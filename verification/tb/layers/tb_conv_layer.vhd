library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library std;
use std.env.all;


entity tb_conv_layer is
    generic (
        G_PREFIX          : string   := "features_0";

        G_C_IN            : positive := 1;
        G_C_OUT           : positive := 64;
        G_H_IN            : positive := 64;
        G_W_IN            : positive := 64;
        G_KERNEL          : positive := 5;
        G_C_PAR           : positive := 2;

        -- Selects which group of G_C_PAR output channels is tested.
        G_OUT_GROUP       : natural  := 0;

        G_VECS            : string :=
            "verification/vectors/default/";

        G_GOLDEN_FILE     : string :=
            "verification/results/default/raw_acc/" &
            "features_0_raw_acc.bin";

        G_RESULT_FILE     : string :=
            "verification/results/default/" &
            "features_0_group_0_received_acc.bin";

        -- 0 means output is always ready.
        -- Values >= 2 stall one cycle every G_STALL_PERIOD cycles.
        G_STALL_PERIOD    : natural := 7;

        G_PROGRESS_STEP   : natural := 10;

        G_TIMEOUT_CYCLES  : positive := 20_000_000
    );
end entity tb_conv_layer;


architecture sim of tb_conv_layer is

    constant C_CLK_PERIOD : time := 10 ns;

    constant C_KERNEL_SIZE : positive :=
        G_KERNEL * G_KERNEL;

    constant C_H_OUT : positive :=
        G_H_IN - G_KERNEL + 1;

    constant C_W_OUT : positive :=
        G_W_IN - G_KERNEL + 1;

    constant C_INPUT_COUNT : positive :=
        G_H_IN * G_W_IN * G_C_IN;

    constant C_OUTPUT_WINDOW_COUNT : positive :=
        C_H_OUT * C_W_OUT;

    constant C_WEIGHT_GROUP_SIZE : positive :=
        G_C_PAR * C_KERNEL_SIZE;

    constant C_SELECTED_WEIGHT_COUNT : positive :=
        G_C_IN * C_WEIGHT_GROUP_SIZE;

    constant C_GROUP_FIRST_CHANNEL : natural :=
        G_OUT_GROUP * G_C_PAR;

    constant C_GROUP_LAST_CHANNEL : natural :=
        C_GROUP_FIRST_CHANNEL + G_C_PAR - 1;


    function expected_weight_group_count return positive is
    begin
        if G_C_IN = 1 then
            -- For a single input channel, the DUT loads one group and
            -- reuses it for every spatial window.
            return 1;
        else
            -- For multiple input channels, every spatial window needs
            -- one streamed weight group per input channel.
            return C_OUTPUT_WINDOW_COUNT * G_C_IN;
        end if;
    end function;


    constant C_EXPECTED_WEIGHT_GROUP_COUNT : positive :=
        expected_weight_group_count;

    constant C_EXPECTED_WEIGHT_TRANSFER_COUNT : positive :=
        C_EXPECTED_WEIGHT_GROUP_COUNT * C_WEIGHT_GROUP_SIZE;


    type t_binary_file is file of character;

    type t_selected_weight_memory is array (
        0 to C_SELECTED_WEIGHT_COUNT - 1
    ) of std_logic_vector(7 downto 0);


    signal clk   : std_logic := '0';
    signal rst_n : std_logic := '0';

    signal i_valid : std_logic := '0';
    signal i_ready : std_logic;
    signal i_data  : std_logic_vector(7 downto 0) :=
        (others => '0');

    signal i_weight_valid : std_logic := '0';
    signal o_weight_ready : std_logic;
    signal i_weight_data  : std_logic_vector(7 downto 0) :=
        (others => '0');

    signal i_acc_ready : std_logic := '0';
    signal o_acc_valid : std_logic;
    signal o_acc_data  :
        std_logic_vector(G_C_PAR * 32 - 1 downto 0);

    signal activation_accept_count : natural := 0;
    signal weight_accept_count     : natural := 0;
    signal output_accept_count     : natural := 0;

begin

    clk <= not clk after C_CLK_PERIOD / 2;


    configuration_check_process : process
    begin
        assert G_KERNEL > 1
            report
                "tb_conv_layer requires G_KERNEL > 1."
            severity failure;

        assert G_W_IN > G_KERNEL
            report
                "The current conv_layer requires G_W_IN > G_KERNEL."
            severity failure;

        assert G_H_IN >= G_KERNEL
            report
                "The current conv_layer requires G_H_IN >= G_KERNEL."
            severity failure;

        assert G_C_OUT mod G_C_PAR = 0
            report
                "G_C_OUT must be divisible by G_C_PAR."
            severity failure;

        assert C_GROUP_LAST_CHANNEL < G_C_OUT
            report
                "G_OUT_GROUP selects output channels outside G_C_OUT."
            severity failure;

        assert G_STALL_PERIOD = 0 or G_STALL_PERIOD >= 2
            report
                "G_STALL_PERIOD must be 0 or at least 2."
            severity failure;

        wait;
    end process;


    dut : entity work.conv_layer
        generic map (
            G_C_IN    => G_C_IN,
            G_C_OUT   => G_C_PAR,
            G_W_IN    => G_W_IN,
            G_H_IN    => G_H_IN,
            G_C_PAR   => G_C_PAR,
            G_KERNEL  => G_KERNEL,
            G_PADDING => 0,
            G_STRIDE  => 1
        )
        port map (
            clk            => clk,
            rst_n          => rst_n,

            i_valid        => i_valid,
            i_ready        => i_ready,
            i_data         => i_data,

            i_weight_valid => i_weight_valid,
            o_weight_ready => o_weight_ready,
            i_weight_data  => i_weight_data,

            i_acc_ready    => i_acc_ready,
            o_acc_valid    => o_acc_valid,
            o_acc_data     => o_acc_data
        );


    reset_process : process
    begin
        rst_n <= '0';

        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait until falling_edge(clk);

        rst_n <= '1';

        wait;
    end process;


    --------------------------------------------------------------------
    -- Read the existing layer input vector and stream it into the DUT.
    --
    -- Existing input ordering:
    --   row -> column -> input channel
    --------------------------------------------------------------------
    activation_driver_process : process
        file f_input : t_binary_file;

        variable v_char : character;
    begin
        i_valid <= '0';
        i_data  <= (others => '0');

        wait until rst_n = '1';

        file_open(
            f_input,
            G_VECS & G_PREFIX & "_in.bin",
            read_mode
        );

        for input_index in 0 to C_INPUT_COUNT - 1 loop

            assert not endfile(f_input)
                report
                    G_PREFIX &
                    ": input vector ended before byte " &
                    integer'image(input_index)
                severity failure;

            read(f_input, v_char);

            wait until falling_edge(clk);

            i_data <= std_logic_vector(
                to_unsigned(character'pos(v_char), 8)
            );

            i_valid <= '1';

            loop
                wait until rising_edge(clk);
                exit when i_ready = '1';
            end loop;
        end loop;

        wait until falling_edge(clk);

        i_valid <= '0';
        i_data  <= (others => '0');

        assert endfile(f_input)
            report
                G_PREFIX &
                ": input vector contains more than " &
                integer'image(C_INPUT_COUNT) &
                " bytes."
            severity failure;

        file_close(f_input);

        wait;
    end process;


    --------------------------------------------------------------------
    -- Load the selected output-channel group from the existing PyTorch
    -- weight vector, then replay the requested input-channel slice
    -- whenever the DUT raises o_weight_ready.
    --
    -- Existing file ordering:
    --   output channel -> input channel -> kernel row -> kernel column
    --
    -- DUT group ordering:
    --   lane 0 KxK -> lane 1 KxK -> ... -> lane G_C_PAR-1 KxK
    --------------------------------------------------------------------
    weight_driver_process : process
        file f_weights : t_binary_file;

        variable v_weights : t_selected_weight_memory;

        variable v_char       : character;
        variable v_lane       : natural;
        variable v_store_addr : natural;

        variable v_channel  : natural range 0 to G_C_IN - 1 := 0;
        variable v_position :
            natural range 0 to C_WEIGHT_GROUP_SIZE - 1 := 0;

        variable v_drive_addr : natural;
    begin
        i_weight_valid <= '0';
        i_weight_data  <= (others => '0');

        file_open(
            f_weights,
            G_VECS & G_PREFIX & "_weights.bin",
            read_mode
        );

        for output_channel in 0 to G_C_OUT - 1 loop
            for input_channel in 0 to G_C_IN - 1 loop
                for kernel_row in 0 to G_KERNEL - 1 loop
                    for kernel_column in 0 to G_KERNEL - 1 loop

                        assert not endfile(f_weights)
                            report
                                G_PREFIX &
                                ": weight vector ended unexpectedly."
                            severity failure;

                        read(f_weights, v_char);

                        if output_channel >=
                           C_GROUP_FIRST_CHANNEL and
                           output_channel <=
                           C_GROUP_LAST_CHANNEL then

                            v_lane :=
                                output_channel -
                                C_GROUP_FIRST_CHANNEL;

                            v_store_addr :=
                                input_channel *
                                    C_WEIGHT_GROUP_SIZE +
                                v_lane *
                                    C_KERNEL_SIZE +
                                kernel_row *
                                    G_KERNEL +
                                kernel_column;

                            v_weights(v_store_addr) :=
                                std_logic_vector(
                                    to_unsigned(
                                        character'pos(v_char),
                                        8
                                    )
                                );
                        end if;

                    end loop;
                end loop;
            end loop;
        end loop;

        assert endfile(f_weights)
            report
                G_PREFIX &
                ": weight vector contains additional bytes."
            severity failure;

        file_close(f_weights);

        wait until rst_n = '1';

        loop
            wait until falling_edge(clk);

            if o_weight_ready = '1' then
                v_drive_addr :=
                    v_channel *
                        C_WEIGHT_GROUP_SIZE +
                    v_position;

                i_weight_data <=
                    v_weights(v_drive_addr);

                i_weight_valid <= '1';
            else
                i_weight_valid <= '0';
            end if;

            wait until rising_edge(clk);

            if i_weight_valid = '1' and
               o_weight_ready = '1' then

                if v_position =
                   C_WEIGHT_GROUP_SIZE - 1 then

                    v_position := 0;

                    if v_channel = G_C_IN - 1 then
                        v_channel := 0;
                    else
                        v_channel := v_channel + 1;
                    end if;

                else
                    v_position := v_position + 1;
                end if;
            end if;
        end loop;
    end process;


    --------------------------------------------------------------------
    -- Periodic output backpressure.
    --------------------------------------------------------------------
    output_ready_process : process
        variable v_cycle : natural := 0;
    begin
        i_acc_ready <= '0';

        wait until rst_n = '1';

        loop
            wait until falling_edge(clk);

            if G_STALL_PERIOD = 0 then
                i_acc_ready <= '1';
            else
                if v_cycle mod G_STALL_PERIOD =
                   G_STALL_PERIOD - 1 then

                    i_acc_ready <= '0';
                else
                    i_acc_ready <= '1';
                end if;

                v_cycle := v_cycle + 1;
            end if;
        end loop;
    end process;


    --------------------------------------------------------------------
    -- Count accepted transfers.
    --------------------------------------------------------------------
    transfer_monitor_process : process(clk)
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                activation_accept_count <= 0;
                weight_accept_count     <= 0;
                output_accept_count     <= 0;

            else
                if i_valid = '1' and
                   i_ready = '1' then

                    activation_accept_count <=
                        activation_accept_count + 1;
                end if;

                if i_weight_valid = '1' and
                   o_weight_ready = '1' then

                    weight_accept_count <=
                        weight_accept_count + 1;
                end if;

                if o_acc_valid = '1' and
                   i_acc_ready = '1' then

                    output_accept_count <=
                        output_accept_count + 1;
                end if;
            end if;
        end if;
    end process;


    --------------------------------------------------------------------
    -- Verify that valid and data remain stable under backpressure.
    --------------------------------------------------------------------
    output_stability_process : process(clk)
        variable v_was_stalled : boolean := false;

        variable v_held_data :
            std_logic_vector(G_C_PAR * 32 - 1 downto 0) :=
            (others => '0');
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                v_was_stalled := false;
                v_held_data   := (others => '0');

            else
                if v_was_stalled then
                    assert o_acc_valid = '1'
                        report
                            G_PREFIX &
                            ": o_acc_valid cleared while stalled."
                        severity failure;

                    assert o_acc_data = v_held_data
                        report
                            G_PREFIX &
                            ": o_acc_data changed while stalled."
                        severity failure;
                end if;

                if o_acc_valid = '1' and
                   i_acc_ready = '0' then

                    v_was_stalled := true;
                    v_held_data   := o_acc_data;
                else
                    v_was_stalled := false;
                end if;
            end if;
        end if;
    end process;


    --------------------------------------------------------------------
    -- Compare each accepted packed accumulator against the full raw
    -- golden file.
    --
    -- Golden file ordering:
    --   output row -> output column -> output channel
    --
    -- Each value is signed int32 little-endian.
    --------------------------------------------------------------------
    output_capture_process : process
        file f_expected : t_binary_file;
        file f_result   : t_binary_file;

        variable v_char : character;

        variable v_expected_bits :
            std_logic_vector(31 downto 0);

        variable v_expected : signed(31 downto 0);
        variable v_got      : signed(31 downto 0);

        variable v_lane     : natural;
        variable v_received : natural := 0;

        variable v_row      : natural;
        variable v_column   : natural;

        variable v_percent      : natural;
        variable v_next_percent : natural := G_PROGRESS_STEP;


        procedure read_i32_le (
            file p_file : t_binary_file;
            variable p_value : out signed(31 downto 0)
        ) is
            variable v_byte_char : character;

            variable v_bits :
                std_logic_vector(31 downto 0);
        begin
            for byte_index in 0 to 3 loop
                assert not endfile(p_file)
                    report
                        G_PREFIX &
                        ": raw accumulator golden file ended " &
                        "unexpectedly."
                    severity failure;

                read(p_file, v_byte_char);

                v_bits(
                    (byte_index + 1) * 8 - 1
                    downto
                    byte_index * 8
                ) :=
                    std_logic_vector(
                        to_unsigned(
                            character'pos(v_byte_char),
                            8
                        )
                    );
            end loop;

            p_value := signed(v_bits);
        end procedure;


        procedure write_i32_le (
            file p_file : t_binary_file;
            constant p_value : in signed(31 downto 0)
        ) is
            variable v_bits :
                std_logic_vector(31 downto 0);
        begin
            v_bits := std_logic_vector(p_value);

            for byte_index in 0 to 3 loop
                write(
                    p_file,
                    character'val(
                        to_integer(
                            unsigned(
                                v_bits(
                                    (byte_index + 1) * 8 - 1
                                    downto
                                    byte_index * 8
                                )
                            )
                        )
                    )
                );
            end loop;
        end procedure;

    begin
        wait until rst_n = '1';

        file_open(
            f_expected,
            G_GOLDEN_FILE,
            read_mode
        );

        file_open(
            f_result,
            G_RESULT_FILE,
            write_mode
        );

        report
            "START " & G_PREFIX &
            " group=" & integer'image(G_OUT_GROUP) &
            " channels=" &
            integer'image(C_GROUP_FIRST_CHANNEL) &
            ".." &
            integer'image(C_GROUP_LAST_CHANNEL) &
            " Cin=" & integer'image(G_C_IN) &
            " Cout=" & integer'image(G_C_OUT) &
            " HxW=" &
            integer'image(G_H_IN) &
            "x" &
            integer'image(G_W_IN) &
            " K=" & integer'image(G_KERNEL) &
            " Cpar=" & integer'image(G_C_PAR);

        while v_received < C_OUTPUT_WINDOW_COUNT loop
            wait until rising_edge(clk);

            if o_acc_valid = '1' and
               i_acc_ready = '1' then

                v_row :=
                    v_received / C_W_OUT;

                v_column :=
                    v_received mod C_W_OUT;

                -- The golden file contains every output channel.
                -- Read one complete output pixel, comparing only the
                -- output channels selected by this simulation.
                for output_channel in 0 to G_C_OUT - 1 loop

                    read_i32_le(
                        f_expected,
                        v_expected
                    );

                    if output_channel >=
                       C_GROUP_FIRST_CHANNEL and
                       output_channel <=
                       C_GROUP_LAST_CHANNEL then

                        v_lane :=
                            output_channel -
                            C_GROUP_FIRST_CHANNEL;

                        v_got :=
                            signed(
                                o_acc_data(
                                    (v_lane + 1) * 32 - 1
                                    downto
                                    v_lane * 32
                                )
                            );

                        write_i32_le(
                            f_result,
                            v_got
                        );

                        assert v_got = v_expected
                            report
                                G_PREFIX &
                                " mismatch:" &
                                " group=" &
                                integer'image(G_OUT_GROUP) &
                                " row=" &
                                integer'image(v_row) &
                                " col=" &
                                integer'image(v_column) &
                                " output_channel=" &
                                integer'image(output_channel) &
                                " expected=" &
                                integer'image(
                                    to_integer(v_expected)
                                ) &
                                " got=" &
                                integer'image(
                                    to_integer(v_got)
                                )
                            severity failure;
                    end if;
                end loop;

                v_received := v_received + 1;

                if G_PROGRESS_STEP > 0 then
                    v_percent :=
                        (v_received * 100) /
                        C_OUTPUT_WINDOW_COUNT;

                    if v_percent >= v_next_percent and
                       v_received <
                       C_OUTPUT_WINDOW_COUNT then

                        report
                            "PROGRESS " &
                            G_PREFIX &
                            " group=" &
                            integer'image(G_OUT_GROUP) &
                            " " &
                            integer'image(v_percent) &
                            "% " &
                            integer'image(v_received) &
                            "/" &
                            integer'image(
                                C_OUTPUT_WINDOW_COUNT
                            ) &
                            " windows";

                        while v_next_percent <=
                              v_percent loop

                            v_next_percent :=
                                v_next_percent +
                                G_PROGRESS_STEP;
                        end loop;
                    end if;
                end if;
            end if;
        end loop;

        assert endfile(f_expected)
            report
                G_PREFIX &
                ": raw accumulator golden file contains " &
                "additional values."
            severity failure;

        file_close(f_expected);
        file_close(f_result);

        -- Allow the final handshake and controller transition to settle.
        for cycle in 1 to 10 loop
            wait until rising_edge(clk);
        end loop;

        assert activation_accept_count =
               C_INPUT_COUNT
            report
                G_PREFIX &
                ": expected " &
                integer'image(C_INPUT_COUNT) &
                " accepted activations, got " &
                integer'image(
                    activation_accept_count
                )
            severity failure;

        assert weight_accept_count =
               C_EXPECTED_WEIGHT_TRANSFER_COUNT
            report
                G_PREFIX &
                ": expected " &
                integer'image(
                    C_EXPECTED_WEIGHT_TRANSFER_COUNT
                ) &
                " accepted weights, got " &
                integer'image(weight_accept_count)
            severity failure;

        assert output_accept_count =
               C_OUTPUT_WINDOW_COUNT
            report
                G_PREFIX &
                ": expected " &
                integer'image(
                    C_OUTPUT_WINDOW_COUNT
                ) &
                " accepted output windows, got " &
                integer'image(output_accept_count)
            severity failure;

        assert o_acc_valid = '0'
            report
                G_PREFIX &
                ": unexpected accumulator remained pending."
            severity failure;

        report
            "PASS " &
            G_PREFIX &
            " group=" &
            integer'image(G_OUT_GROUP) &
            " channels=" &
            integer'image(C_GROUP_FIRST_CHANNEL) &
            ".." &
            integer'image(C_GROUP_LAST_CHANNEL) &
            " windows=" &
            integer'image(C_OUTPUT_WINDOW_COUNT)
            severity note;

        stop(0);
        wait;
    end process;


    watchdog_process : process
    begin
        for cycle in 1 to G_TIMEOUT_CYCLES loop
            wait until rising_edge(clk);
        end loop;

        assert false
            report
                "TIMEOUT " &
                G_PREFIX &
                " group=" &
                integer'image(G_OUT_GROUP)
            severity failure;
    end process;

end architecture sim;