library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.env.all;

entity tb_conv_layer is
    generic (
        G_PREFIX   : string   := "features_0";
        G_C_IN     : positive := 1;
        G_C_OUT    : positive := 64;
        G_H_IN     : positive := 64;
        G_W_IN     : positive := 64;
        G_KERNEL   : positive := 5;
        G_PADDING  : natural  := 2;
        G_PAR_MACS : positive := 2;
        G_VECS     : string   := "verification/vectors/default/";
        G_RESS     : string   := "verification/results/default/";
        G_PROGRESS_STEP : natural := 10
    );
end entity tb_conv_layer;

architecture sim of tb_conv_layer is

    -- Derived sizes used for loading and output counting
    constant C_W_DEPTH  : positive :=
        G_C_OUT / G_PAR_MACS * G_KERNEL * G_KERNEL * G_C_IN;
    constant C_N_IN     : positive := G_H_IN * G_W_IN * G_C_IN;
    constant C_N_OUT    : positive := G_H_IN * G_W_IN * G_C_OUT;

    -- Vector file paths (relative to project root where sim is invoked)
    constant C_VECS : string := G_VECS;
    constant C_RESS : string := G_RESS;

    -- Clock / reset
    signal clk   : std_logic := '0';
    signal rst_n : std_logic := '0';

    -- Streaming interface
    signal i_valid : std_logic := '0';
    signal i_ready : std_logic;
    signal i_data  : std_logic_vector(7 downto 0) := (others => '0');
    signal i_last  : std_logic := '0';
    signal o_valid : std_logic;
    signal o_ready : std_logic := '1';
    signal o_data  : std_logic_vector(7 downto 0);
    signal o_last  : std_logic;

    -- Config port
    signal cfg_we    : std_logic                     := '0';
    signal cfg_sel   : std_logic_vector(1 downto 0)  := (others => '0');
    signal cfg_addr  : std_logic_vector(19 downto 0) := (others => '0');
    signal cfg_wdata : std_logic_vector(31 downto 0) := (others => '0');

begin

    clk <= not clk after 5 ns;  -- 100 MHz

    dut : entity work.conv_layer
        generic map (
            G_C_IN     => G_C_IN,
            G_C_OUT    => G_C_OUT,
            G_H_IN     => G_H_IN,
            G_W_IN     => G_W_IN,
            G_KERNEL   => G_KERNEL,
            G_PADDING  => G_PADDING,
            G_PAR_MACS => G_PAR_MACS
        )
        port map (
            clk      => clk,
            rst_n    => rst_n,
            i_valid  => i_valid,
            i_ready  => i_ready,
            i_data   => i_data,
            i_last   => i_last,
            o_valid  => o_valid,
            o_ready  => o_ready,
            o_data   => o_data,
            o_last   => o_last,
            cfg_we   => cfg_we,
            cfg_sel  => cfg_sel,
            cfg_addr => cfg_addr,
            cfg_wdata=> cfg_wdata
        );

    -- ---------------------------------------------------------------
    -- p_load: read parameter binaries and write through cfg port
    --         runs before rst_n is released
    -- ---------------------------------------------------------------
    p_load : process
        type bin_file is file of character;
        file     f       : bin_file;
        variable ch      : character;
        variable b0, b1, b2, b3 : integer;
        variable v_slv   : std_logic_vector(31 downto 0);
        variable v_byte  : integer;
        variable v_addr  : integer;

        procedure cfg_write (
            sel  : in std_logic_vector(1 downto 0);
            addr : in integer;
            data : in std_logic_vector(31 downto 0)) is
        begin
            cfg_sel   <= sel;
            cfg_addr  <= std_logic_vector(to_unsigned(addr, 20));
            cfg_wdata <= data;
            cfg_we    <= '1';
            wait until rising_edge(clk);
            cfg_we    <= '0';
        end procedure;

    begin
        report "START " & G_PREFIX &
               "  Cin=" & integer'image(G_C_IN) &
               "  Cout=" & integer'image(G_C_OUT) &
               "  HxW=" & integer'image(G_H_IN) & "x" & integer'image(G_W_IN) &
               "  K=" & integer'image(G_KERNEL) &
               "  P=" & integer'image(G_PADDING) &
               "  PAR=" & integer'image(G_PAR_MACS);

        -- weights: PyTorch order C_out x C_in x KH x KW
        --   -> DUT flat addr = (c_out/PAR_MACS)*K*K*C_IN + kh*K*C_IN + kw*C_IN + c_in
        --     bank = c_out mod PAR_MACS
        --     flat cfg_addr = bank * C_W_DEPTH + position
        file_open(f, C_VECS & G_PREFIX & "_weights.bin", read_mode);
        for c_out in 0 to G_C_OUT - 1 loop
            for c_in in 0 to G_C_IN - 1 loop
                for kh in 0 to G_KERNEL - 1 loop
                    for kw in 0 to G_KERNEL - 1 loop
                        read(f, ch);
                        v_byte := character'pos(ch);
                        -- sign-extend uint8 -> int8
                        if v_byte >= 128 then v_byte := v_byte - 256; end if;
                        v_addr :=
                            (c_out mod G_PAR_MACS) * C_W_DEPTH +
                            (c_out / G_PAR_MACS) * G_KERNEL * G_KERNEL * G_C_IN +
                            kh * G_KERNEL * G_C_IN +
                            kw * G_C_IN + c_in;
                        v_slv := std_logic_vector(to_signed(v_byte, 32));
                        cfg_write("00", v_addr, v_slv);
                    end loop;
                end loop;
            end loop;
        end loop;
        file_close(f);

        -- biases: int32 little-endian, one per output channel
        file_open(f, C_VECS & G_PREFIX & "_biases.bin", read_mode);
        for c_out in 0 to G_C_OUT - 1 loop
            read(f, ch); b0 := character'pos(ch);
            read(f, ch); b1 := character'pos(ch);
            read(f, ch); b2 := character'pos(ch);
            read(f, ch); b3 := character'pos(ch);
            v_slv(7  downto  0) := std_logic_vector(to_unsigned(b0, 8));
            v_slv(15 downto  8) := std_logic_vector(to_unsigned(b1, 8));
            v_slv(23 downto 16) := std_logic_vector(to_unsigned(b2, 8));
            v_slv(31 downto 24) := std_logic_vector(to_unsigned(b3, 8));
            cfg_write("01", c_out, v_slv);
        end loop;
        file_close(f);

        -- rq_m: uint32 little-endian
        file_open(f, C_VECS & G_PREFIX & "_requant_m.bin", read_mode);
        for c_out in 0 to G_C_OUT - 1 loop
            read(f, ch); b0 := character'pos(ch);
            read(f, ch); b1 := character'pos(ch);
            read(f, ch); b2 := character'pos(ch);
            read(f, ch); b3 := character'pos(ch);
            v_slv(7  downto  0) := std_logic_vector(to_unsigned(b0, 8));
            v_slv(15 downto  8) := std_logic_vector(to_unsigned(b1, 8));
            v_slv(23 downto 16) := std_logic_vector(to_unsigned(b2, 8));
            v_slv(31 downto 24) := std_logic_vector(to_unsigned(b3, 8));
            cfg_write("10", c_out, v_slv);
        end loop;
        file_close(f);

        -- rq_r: uint8, one per output channel
        file_open(f, C_VECS & G_PREFIX & "_requant_r.bin", read_mode);
        for c_out in 0 to G_C_OUT - 1 loop
            read(f, ch);
            v_slv := std_logic_vector(to_unsigned(character'pos(ch), 32));
            cfg_write("11", c_out, v_slv);
        end loop;
        file_close(f);

        -- release reset after all parameters are loaded
        wait until rising_edge(clk);
        rst_n <= '1';
        wait;
    end process p_load;

    -- ---------------------------------------------------------------
    -- p_stim: stream layer input into the DUT byte by byte
    -- ---------------------------------------------------------------
    p_stim : process
        type bin_file is file of character;
        file     f_in   : bin_file;
        variable ch     : character;
        variable n_sent : integer := 0;
    begin
        -- wait for reset release (driven by p_load)
        wait until rst_n = '1';
        wait until rising_edge(clk);

        file_open(f_in, C_VECS & G_PREFIX & "_in.bin", read_mode);

        while not endfile(f_in) loop
            read(f_in, ch);
            n_sent := n_sent + 1;

            i_data  <= std_logic_vector(to_unsigned(character'pos(ch), 8));
            i_valid <= '1';
            if n_sent = C_N_IN then
                i_last <= '1';
            end if;

            -- wait until the transfer is accepted
            loop
                wait until rising_edge(clk);
                exit when i_ready = '1';
            end loop;
        end loop;

        i_valid <= '0';
        i_last  <= '0';
        file_close(f_in);
        wait;
    end process p_stim;

    -- ---------------------------------------------------------------
    -- p_capture: collect output, compare byte-exact against expected
    -- ---------------------------------------------------------------
    p_capture : process
        type bin_file is file of character;
        file     f_exp  : bin_file;
        file     f_res  : bin_file;
        variable ch_exp : character;
        variable got    : integer;
        variable exp    : integer;
        variable n_recv : integer := 0;
        variable pct    : integer;
        variable next_pct : integer := G_PROGRESS_STEP;
        variable pixel  : integer;
        variable row    : integer;
        variable col    : integer;
        variable ch     : integer;
    begin
        wait until rst_n = '1';

        file_open(f_exp, C_VECS & G_PREFIX & "_out.bin", read_mode);
        file_open(f_res, C_RESS & G_PREFIX & "_conv_layer_out.bin", write_mode);

        while n_recv < C_N_OUT loop
            wait until rising_edge(clk);
            if o_valid = '1' and o_ready = '1' then
                read(f_exp, ch_exp);
                got := to_integer(unsigned(o_data));
                exp := character'pos(ch_exp);

                write(f_res, character'val(got));

                pixel := n_recv / G_C_OUT;
                row   := pixel / G_W_IN;
                col   := pixel mod G_W_IN;
                ch    := n_recv mod G_C_OUT;
                pct   := (n_recv * 100) / C_N_OUT;

                assert got = exp
                    report G_PREFIX & " MISMATCH at byte " & integer'image(n_recv) &
                           "  row="      & integer'image(row) &
                           "  col="      & integer'image(col) &
                           "  ch="       & integer'image(ch) &
                           "  pct="      & integer'image(pct) & "%" &
                           "  expected=" & integer'image(exp) &
                           "  got="      & integer'image(got)
                    severity failure;

                n_recv := n_recv + 1;

                if G_PROGRESS_STEP > 0 then
                    pct := (n_recv * 100) / C_N_OUT;
                    if pct >= next_pct and n_recv < C_N_OUT then
                        report "PROGRESS " & G_PREFIX & " " &
                               integer'image(pct) & "%  " &
                               integer'image(n_recv) & "/" &
                               integer'image(C_N_OUT) & " bytes";

                        while next_pct <= pct loop
                            next_pct := next_pct + G_PROGRESS_STEP;
                        end loop;
                    end if;
                end if;
            end if;
        end loop;

        file_close(f_exp);
        file_close(f_res);

        report "PASS " & G_PREFIX & " - all " & integer'image(C_N_OUT) &
               " output bytes match";
        stop(0);
    end process p_capture;

    -- ---------------------------------------------------------------
    -- Timeout watchdog - fails if simulation hangs
    -- ---------------------------------------------------------------
    p_watchdog : process
    begin
        wait for 100 ms;
        assert false report "TIMEOUT" severity failure;
    end process p_watchdog;

end architecture sim;
