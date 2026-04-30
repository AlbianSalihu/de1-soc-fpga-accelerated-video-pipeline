library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity conv_layer is
    generic (
        G_C_IN     : positive;
        G_C_OUT    : positive;
        G_H_IN     : positive;
        G_W_IN     : positive;
        G_KERNEL   : positive := 3;
        G_PADDING  : natural  := 1;
        G_PAR_MACS : positive;
        G_USE_BRAM : boolean  := true;
        G_USE_REGS : boolean  := false
    );
    port (
        clk     : in  std_logic;
        rst_n   : in  std_logic;

        -- streaming input
        i_valid : in  std_logic;
        i_ready : out std_logic;
        i_data  : in  std_logic_vector(7 downto 0);
        i_last  : in  std_logic;

        -- streaming output
        o_valid : out std_logic;
        o_ready : in  std_logic;
        o_data  : out std_logic_vector(7 downto 0);
        o_last  : out std_logic;

        -- parameter load port (driven by pipeline_ctrl in V2; testbench in sim)
        -- one write per cycle; 32-bit data covers all entry widths
        cfg_we    : in  std_logic;
        cfg_sel   : in  std_logic_vector(1 downto 0); -- 00=weights 01=bias 10=rq_m 11=rq_r
        cfg_addr  : in  std_logic_vector(19 downto 0);
        cfg_wdata : in  std_logic_vector(31 downto 0)
    );
end entity conv_layer;

architecture rtl of conv_layer is

    constant C_ROW_LAST : natural := G_H_IN + G_PADDING - 1;
    constant C_COL_LAST : natural := G_W_IN + G_PADDING - 1;

    -- FSM
    type t_state is (S_INIT, S_STEADY, S_COMPUTE, S_FLUSH);
    signal state : t_state;

    -- S_COMPUTE sub-phases
    type t_compute_phase is (ACCUMULATE, OUTPUT);
    signal compute_phase : t_compute_phase;

    -- Counters
    signal row_cnt : integer range 0 to C_ROW_LAST;
    signal col_cnt : integer range 0 to C_COL_LAST;
    signal ch_cnt  : integer range 0 to G_C_IN - 1;
    signal out_grp : integer range 0 to G_C_OUT / G_PAR_MACS - 1;
    signal out_byte: integer range 0 to G_PAR_MACS - 1;

    -- Window position counters used during S_COMPUTE ACCUMULATE phase
    -- Together they walk every (row, col, ch) combination in the KxKxC_IN window
    signal win_row : integer range 0 to G_KERNEL - 1;
    signal win_col : integer range 0 to G_KERNEL - 1;
    signal win_ch  : integer range 0 to G_C_IN - 1;
    -- flat MAC counter - weight BRAM address within the current output group
    signal mac_cnt : integer range 0 to G_KERNEL * G_KERNEL * G_C_IN - 1;

    -- Window valid flag (combinational)
    signal window_valid : std_logic;

    -- Line buffers: K rows x (W * C_IN) bytes.  The current row is written
    -- immediately, so the circular store needs K slots to avoid overwriting
    -- the oldest row while it is still part of a KxK padded window.
    -- Each position stores one byte - col and channel are flattened into one index
    -- Tool infers BRAM or registers based on G_USE_REGS
    type t_line_buf is array (0 to G_KERNEL - 1, 0 to G_W_IN * G_C_IN - 1) of std_logic_vector(7 downto 0);
    signal line_buf : t_line_buf;

    -- Write row pointer for line buffers (circular)
    signal lb_wr_row   : integer range 0 to G_KERNEL - 1;
    -- Read start pointer - which slot is the oldest row (circular)
    signal lb_rd_start : integer range 0 to G_KERNEL - 1;

    -- Shift register: K pixels wide, one complete pixel = C_IN bytes per slot
    -- Holds the current incoming row, K pixels deep
    type t_shift is array (0 to G_KERNEL - 1) of std_logic_vector(G_C_IN * 8 - 1 downto 0);
    signal shift_reg : t_shift;

    -- G_PAR_MACS int32 accumulators (initialised to bias, then accumulated into)
    type t_acc is array (0 to G_PAR_MACS - 1) of signed(31 downto 0);
    signal acc : t_acc;

    -- ---------------------------------------------------------------
    -- Weight, bias and requantisation parameter storage
    -- All three are written once at load time (testbench / config bus).
    -- Declared as plain arrays; synthesis infers BRAM for large layers.
    -- ---------------------------------------------------------------

    -- Weight banks: G_PAR_MACS independent banks so all parallel MACs read
    -- in a single cycle.  Address = out_grp * K*K*C_IN + mac_cnt.
    -- Weights stored in win_row x win_col x win_ch iteration order per channel.
    constant C_W_DEPTH : natural :=
        G_C_OUT / G_PAR_MACS * G_KERNEL * G_KERNEL * G_C_IN;
    type t_weight_bank  is array (0 to C_W_DEPTH - 1) of signed(7 downto 0);
    type t_weight_banks is array (0 to G_PAR_MACS - 1) of t_weight_bank;
    signal weight_banks : t_weight_banks;

    -- One int32 bias per output channel
    type t_bias_store is array (0 to G_C_OUT - 1) of signed(31 downto 0);
    signal bias_store : t_bias_store;

    -- Per-channel fixed-point requant parameters (written by testbench / ctrl)
    type t_rq_m is array (0 to G_C_OUT - 1) of unsigned(31 downto 0);
    type t_rq_r is array (0 to G_C_OUT - 1) of unsigned(7 downto 0);
    signal rq_m : t_rq_m;
    signal rq_r : t_rq_r;

    -- Registered ready signal (avoids combinational loop on i_ready)
    signal ready_r : std_logic;

begin

    i_ready <= ready_r;

    -- Window is valid when enough rows and columns are buffered.
    -- Padding fills missing border positions with zero so the window
    -- becomes valid G_PADDING positions earlier in each dimension.
    window_valid <= '1' when
        row_cnt >= G_KERNEL - 1 - G_PADDING and
        col_cnt >= G_KERNEL - 1 - G_PADDING
        else '0';

    -- ---------------------------------------------------------------
    -- FSM
    -- ---------------------------------------------------------------
    process(clk)
        -- ACCUMULATE helpers
        variable v_act_row  : integer;
        variable v_act_col  : integer;
        variable v_lb_slot  : integer range 0 to G_KERNEL - 1;
        variable v_act_byte : signed(8 downto 0);
        variable v_w_addr   : integer range 0 to C_W_DEPTH - 1;
        -- OUTPUT / requant helpers
        variable v_c_out    : integer range 0 to G_C_OUT - 1;
        variable v_acc_pos  : signed(31 downto 0);   -- ReLU'd accumulator
        variable v_prod     : unsigned(63 downto 0); -- acc_pos * rq_m
        variable v_shifted  : unsigned(63 downto 0); -- after rounding right-shift
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                state             <= S_INIT;
                compute_phase     <= ACCUMULATE;
                row_cnt           <= 0;
                col_cnt           <= 0;
                ch_cnt            <= 0;
                out_grp           <= 0;
                out_byte          <= 0;
                win_row           <= 0;
                win_col           <= 0;
                win_ch            <= 0;
                mac_cnt           <= 0;
                lb_wr_row         <= 0;
                lb_rd_start <= 0;
                ready_r     <= '1';
                o_valid           <= '0';
                o_last            <= '0';
            else
                case state is

                    when S_INIT =>
                        o_valid <= '0';
                        o_last  <= '0';

                        if i_valid = '1' and ready_r = '1' then
                            line_buf(lb_wr_row, col_cnt * G_C_IN + ch_cnt) <= i_data;

                            -- shift array left at the START of each new pixel (ch_cnt=0)
                            -- so shift_reg holds the K most recent pixels of the current row.
                            if ch_cnt = 0 then
                                for k in 0 to G_KERNEL - 2 loop
                                    shift_reg(k) <= shift_reg(k + 1);
                                end loop;
                            end if;
                            -- pack every byte into the rightmost slot as it arrives
                            shift_reg(G_KERNEL - 1)(
                                (G_C_IN - ch_cnt) * 8 - 1 downto
                                (G_C_IN - ch_cnt - 1) * 8
                            ) <= i_data;

                            if ch_cnt = G_C_IN - 1 then
                                ch_cnt <= 0;

                                if window_valid = '1' then
                                    -- Padding makes the first output valid before K-1
                                    -- complete input rows/columns have arrived.
                                    ready_r       <= '0';
                                    state         <= S_COMPUTE;
                                    out_grp       <= 0;
                                    win_row       <= 0;
                                    win_col       <= 0;
                                    win_ch        <= 0;
                                    mac_cnt       <= 0;
                                    compute_phase <= ACCUMULATE;
                                    for m in 0 to G_PAR_MACS - 1 loop
                                        acc(m) <= bias_store(m);
                                    end loop;

                                elsif col_cnt = G_W_IN - 1 then
                                    col_cnt <= 0;
                                    row_cnt <= row_cnt + 1;

                                    if lb_wr_row = G_KERNEL - 1 then
                                        lb_wr_row <= 0;
                                    else
                                        lb_wr_row <= lb_wr_row + 1;
                                    end if;

                                    if lb_rd_start = G_KERNEL - 1 then
                                        lb_rd_start <= 0;
                                    else
                                        lb_rd_start <= lb_rd_start + 1;
                                    end if;

                                else
                                    col_cnt <= col_cnt + 1;
                                end if;

                            else
                                ch_cnt <= ch_cnt + 1;
                            end if;
                        end if;

                    when S_STEADY =>
                        -- Ready is asserted only while the current coordinate maps to
                        -- a real input pixel.  Right/bottom padding coordinates are
                        -- generated internally with upstream stalled.
                        if row_cnt < G_H_IN and col_cnt < G_W_IN then
                            ready_r <= '1';
                        else
                            ready_r <= '0';
                        end if;
                        o_valid <= '0';
                        o_last  <= '0';

                        if row_cnt < G_H_IN and col_cnt < G_W_IN and i_valid = '1' then
                            line_buf(lb_wr_row, col_cnt * G_C_IN + ch_cnt) <= i_data;

                            -- shift array left at start of each new pixel (ch_cnt=0)
                            if ch_cnt = 0 then
                                for k in 0 to G_KERNEL - 2 loop
                                    shift_reg(k) <= shift_reg(k + 1);
                                end loop;
                            end if;
                            -- pack every incoming byte into rightmost slot
                            shift_reg(G_KERNEL - 1)(
                                (G_C_IN - ch_cnt) * 8 - 1 downto
                                (G_C_IN - ch_cnt - 1) * 8
                            ) <= i_data;

                            if ch_cnt = G_C_IN - 1 then
                                ch_cnt <= 0;

                                if window_valid = '1' then
                                    -- full pixel received, window complete -> stall and compute
                                    ready_r       <= '0';
                                    state         <= S_COMPUTE;
                                    out_grp       <= 0;
                                    win_row       <= 0;
                                    win_col       <= 0;
                                    win_ch        <= 0;
                                    mac_cnt       <= 0;
                                    compute_phase <= ACCUMULATE;
                                    -- pre-load accumulators with bias for out_grp 0
                                    for m in 0 to G_PAR_MACS - 1 loop
                                        acc(m) <= bias_store(m);
                                    end loop;

                                elsif i_last = '1' then
                                    state <= S_FLUSH;

                                else
                                    -- window not yet valid (left margin); advance position
                                    if col_cnt = G_W_IN - 1 then
                                        col_cnt <= 0;
                                        row_cnt <= row_cnt + 1;
                                        if lb_wr_row = G_KERNEL - 1 then
                                            lb_wr_row <= 0;
                                        else
                                            lb_wr_row <= lb_wr_row + 1;
                                        end if;
                                        if lb_rd_start = G_KERNEL - 1 then
                                            lb_rd_start <= 0;
                                        else
                                            lb_rd_start <= lb_rd_start + 1;
                                        end if;
                                    else
                                        col_cnt <= col_cnt + 1;
                                    end if;
                                end if;

                            else
                                ch_cnt <= ch_cnt + 1;
                            end if;

                        elsif row_cnt >= G_H_IN or col_cnt >= G_W_IN then
                            -- Virtual zero pixel for right/bottom padding.
                            ch_cnt <= 0;
                            if ch_cnt = 0 then
                                for k in 0 to G_KERNEL - 2 loop
                                    shift_reg(k) <= shift_reg(k + 1);
                                end loop;
                            end if;
                            shift_reg(G_KERNEL - 1) <= (others => '0');

                            if window_valid = '1' then
                                ready_r       <= '0';
                                state         <= S_COMPUTE;
                                out_grp       <= 0;
                                win_row       <= 0;
                                win_col       <= 0;
                                win_ch        <= 0;
                                mac_cnt       <= 0;
                                compute_phase <= ACCUMULATE;
                                for m in 0 to G_PAR_MACS - 1 loop
                                    acc(m) <= bias_store(m);
                                end loop;
                            elsif col_cnt = C_COL_LAST then
                                col_cnt <= 0;
                                if row_cnt < C_ROW_LAST then
                                    row_cnt <= row_cnt + 1;
                                end if;
                            else
                                col_cnt <= col_cnt + 1;
                            end if;
                        end if;

                    when S_COMPUTE =>
                        if compute_phase = ACCUMULATE then
                            o_valid <= '0';
                            o_last  <= '0';

                            -- -- activation byte for this MAC cycle ------------------
                            v_act_row := row_cnt - (G_KERNEL - 1) + win_row;
                            v_act_col := col_cnt - (G_KERNEL - 1) + win_col;

                            if v_act_row < 0 or v_act_row >= G_H_IN or
                               v_act_col < 0 or v_act_col >= G_W_IN then
                                v_act_byte := (others => '0');          -- zero padding
                            elsif win_row = G_KERNEL - 1 then
                                -- current row: shift register
                                v_act_byte := signed('0' & shift_reg(win_col)(
                                    (G_C_IN - win_ch) * 8 - 1 downto
                                    (G_C_IN - win_ch - 1) * 8));
                            else
                                -- older rows: each input row is stored in row mod K.
                                v_lb_slot := v_act_row mod G_KERNEL;
                                v_act_byte := signed('0' &
                                    line_buf(v_lb_slot, v_act_col * G_C_IN + win_ch));
                            end if;

                            -- -- multiply-accumulate: G_PAR_MACS channels in parallel ─
                            v_w_addr := out_grp * G_KERNEL * G_KERNEL * G_C_IN + mac_cnt;
                            for m in 0 to G_PAR_MACS - 1 loop
                                acc(m) <= acc(m) +
                                    resize(v_act_byte * weight_banks(m)(v_w_addr), 32);
                            end loop;

                            -- advance counters: ch innermost, row outermost
                            if win_ch = G_C_IN - 1 then
                                win_ch <= 0;
                                if win_col = G_KERNEL - 1 then
                                    win_col <= 0;
                                    if win_row = G_KERNEL - 1 then
                                        -- all K*K*C_IN elements done: reset for next out_grp
                                        win_row       <= 0;
                                        mac_cnt       <= 0;
                                        compute_phase <= OUTPUT;
                                        out_byte      <= 0;
                                    else
                                        win_row <= win_row + 1;
                                        mac_cnt <= mac_cnt + 1;
                                    end if;
                                else
                                    win_col <= win_col + 1;
                                    mac_cnt <= mac_cnt + 1;
                                end if;
                            else
                                win_ch  <= win_ch + 1;
                                mac_cnt <= mac_cnt + 1;
                            end if;

                        else  -- -- OUTPUT phase ------------------------------------─

                            -- requant: clip( round( relu(acc) * rq_m >> rq_r ), 0, 255 )
                            v_c_out := out_grp * G_PAR_MACS + out_byte;

                            -- ReLU: clamp negative accumulator to zero
                            if acc(out_byte)(31) = '1' then
                                v_acc_pos := (others => '0');
                            else
                                v_acc_pos := acc(out_byte);
                            end if;

                            -- 32x32->64 bit unsigned multiply (acc_pos >= 0, rq_m is unsigned)
                            v_prod := unsigned(std_logic_vector(v_acc_pos)) * rq_m(v_c_out);

                            -- rounding: add 2^(r-1) before shifting (skip when r=0)
                            if rq_r(v_c_out) /= 0 then
                                v_prod := v_prod +
                                    shift_left(to_unsigned(1, 64),
                                               to_integer(rq_r(v_c_out)) - 1);
                            end if;

                            v_shifted := shift_right(v_prod, to_integer(rq_r(v_c_out)));

                            -- saturate to uint8 and drive output
                            if v_shifted > 255 then
                                o_data <= (others => '1');
                            else
                                o_data <= std_logic_vector(v_shifted(7 downto 0));
                            end if;
                            o_valid <= '1';

                            -- o_last on the very last byte of the output frame
                            if out_grp  = G_C_OUT / G_PAR_MACS - 1 and
                               out_byte = G_PAR_MACS - 1             and
                               row_cnt  = C_ROW_LAST                 and
                               col_cnt  = C_COL_LAST then
                                o_last <= '1';
                            else
                                o_last <= '0';
                            end if;

                            if o_ready = '1' then
                                if out_byte = G_PAR_MACS - 1 then
                                    out_byte <= 0;

                                    if out_grp = G_C_OUT / G_PAR_MACS - 1 then
                                        -- -- all groups done: advance position --
                                        out_grp <= 0;

                                        if row_cnt = C_ROW_LAST and col_cnt = C_COL_LAST then
                                            ready_r <= '1';
                                            state   <= S_FLUSH;
                                        else
                                            state <= S_STEADY;

                                            if col_cnt = C_COL_LAST then
                                                col_cnt <= 0;
                                                row_cnt <= row_cnt + 1;
                                                if row_cnt < G_H_IN - 1 then
                                                    if lb_wr_row = G_KERNEL - 1 then
                                                        lb_wr_row <= 0;
                                                    else
                                                        lb_wr_row <= lb_wr_row + 1;
                                                    end if;
                                                    if lb_rd_start = G_KERNEL - 1 then
                                                        lb_rd_start <= 0;
                                                    else
                                                        lb_rd_start <= lb_rd_start + 1;
                                                    end if;
                                                end if;

                                                if row_cnt + 1 < G_H_IN then
                                                    ready_r <= '1';
                                                else
                                                    ready_r <= '0';
                                                end if;
                                            else
                                                col_cnt <= col_cnt + 1;
                                                if row_cnt < G_H_IN and col_cnt + 1 < G_W_IN then
                                                    ready_r <= '1';
                                                else
                                                    ready_r <= '0';
                                                end if;
                                            end if;
                                        end if;

                                    else
                                        -- -- next output group: load bias, restart MAC walk --─
                                        out_grp       <= out_grp + 1;
                                        compute_phase <= ACCUMULATE;
                                        win_row       <= 0;
                                        win_col       <= 0;
                                        win_ch        <= 0;
                                        mac_cnt       <= 0;
                                        for m in 0 to G_PAR_MACS - 1 loop
                                            acc(m) <= bias_store(
                                                (out_grp + 1) * G_PAR_MACS + m);
                                        end loop;
                                    end if;

                                else
                                    out_byte <= out_byte + 1;
                                end if;
                            end if;

                        end if;

                    when S_FLUSH =>
                        -- reset all datapath counters and pointers; weights/biases stay loaded
                        state         <= S_INIT;
                        compute_phase <= ACCUMULATE;
                        row_cnt       <= 0;
                        col_cnt       <= 0;
                        ch_cnt        <= 0;
                        out_grp       <= 0;
                        out_byte      <= 0;
                        win_row       <= 0;
                        win_col       <= 0;
                        win_ch        <= 0;
                        mac_cnt       <= 0;
                        lb_wr_row     <= 0;
                        lb_rd_start   <= 0;
                        ready_r       <= '1';
                        o_valid       <= '0';
                        o_last        <= '0';
                        for m in 0 to G_PAR_MACS - 1 loop
                            acc(m) <= (others => '0');
                        end loop;

                end case;
            end if;
        end if;
    end process;

    -- ---------------------------------------------------------------
    -- Parameter load process
    -- Writes weight_banks, bias_store, rq_m, rq_r on cfg_we='1'.
    -- cfg_sel selects the target store; cfg_addr is the flat index within it.
    -- For weights: addr = mac_unit * C_W_DEPTH + position (flat)
    -- For bias/rq_m/rq_r: addr = output channel index
    -- ---------------------------------------------------------------
    process(clk)
        variable v_addr : integer;
    begin
        if rising_edge(clk) then
            if cfg_we = '1' then
                v_addr := to_integer(unsigned(cfg_addr));
                case cfg_sel is
                    when "00" =>  -- weights: lower byte, addr is flat bank index
                        weight_banks(v_addr / C_W_DEPTH)(v_addr mod C_W_DEPTH)
                            <= signed(cfg_wdata(7 downto 0));
                    when "01" =>  -- bias: full 32 bits
                        bias_store(v_addr) <= signed(cfg_wdata);
                    when "10" =>  -- rq_m: full 32 bits
                        rq_m(v_addr) <= unsigned(cfg_wdata);
                    when others =>  -- rq_r: lower byte
                        rq_r(v_addr) <= unsigned(cfg_wdata(7 downto 0));
                end case;
            end if;
        end if;
    end process;

end architecture rtl;
