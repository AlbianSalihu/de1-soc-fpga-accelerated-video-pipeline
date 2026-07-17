library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library std;
use std.env.all;

entity tb_conv_layer_initial_line_fill is
end entity tb_conv_layer_initial_line_fill;

architecture sim of tb_conv_layer_initial_line_fill is

    constant C_CLK_PERIOD : time := 10 ns;

    constant C_C_IN   : positive := 1;
    constant C_W_IN   : positive := 4;
    constant C_KERNEL : positive := 3;

    constant C_LINE_SIZE : positive :=
        C_W_IN * C_C_IN;

    constant C_INITIAL_FILL_SIZE : positive :=
        (C_KERNEL - 1) * C_LINE_SIZE;

    signal clk   : std_logic := '0';
    signal rst_n : std_logic := '0';

    signal i_valid : std_logic := '0';
    signal i_ready : std_logic;
    signal i_data  : std_logic_vector(7 downto 0) := (others => '0');


    procedure send_activation_byte (
        signal p_clk   : in  std_logic;
        signal p_valid : out std_logic;
        signal p_ready : in  std_logic;
        signal p_data  : out std_logic_vector(7 downto 0);
        constant value : in  natural
    ) is
    begin
        p_data  <= std_logic_vector(to_unsigned(value, p_data'length));
        p_valid <= '1';

        -- Wait until the byte is accepted.
        loop
            wait until rising_edge(p_clk);
            exit when p_ready = '1';
        end loop;

        p_valid <= '0';
    end procedure;

begin

    --------------------------------------------------------------------
    -- Clock generation
    --------------------------------------------------------------------

    clk <= not clk after C_CLK_PERIOD / 2;


    --------------------------------------------------------------------
    -- Device under test
    --------------------------------------------------------------------

    dut : entity work.conv_layer
        generic map (
            G_C_IN   => C_C_IN,
            G_W_IN   => C_W_IN,
            G_KERNEL => C_KERNEL
        )
        port map (
            clk     => clk,
            rst_n   => rst_n,

            i_valid => i_valid,
            i_ready => i_ready,
            i_data  => i_data
        );


    --------------------------------------------------------------------
    -- Test sequence
    --------------------------------------------------------------------

    stimulus_process : process
    begin

        ----------------------------------------------------------------
        -- Reset
        ----------------------------------------------------------------

        rst_n   <= '0';
        i_valid <= '0';
        i_data  <= (others => '0');

        wait for 3 * C_CLK_PERIOD;
        wait until falling_edge(clk);

        rst_n <= '1';

        ----------------------------------------------------------------
        -- The first rising edge after reset starts initial line filling.
        ----------------------------------------------------------------

        wait until rising_edge(clk);
        wait for 1 ns;

        assert i_ready = '1'
            report
                "FAIL: i_ready was not asserted when initial line filling started."
            severity failure;


        ----------------------------------------------------------------
        -- Send the first seven bytes.
        --
        -- Expected storage:
        --
        -- line_buffer(0, 0) = 0
        -- line_buffer(0, 1) = 1
        -- line_buffer(0, 2) = 2
        -- line_buffer(0, 3) = 3
        --
        -- line_buffer(1, 0) = 4
        -- line_buffer(1, 1) = 5
        -- line_buffer(1, 2) = 6
        ----------------------------------------------------------------

        for value in 0 to C_INITIAL_FILL_SIZE - 2 loop

            send_activation_byte(
                p_clk   => clk,
                p_valid => i_valid,
                p_ready => i_ready,
                p_data  => i_data,
                value   => value
            );

            wait for 1 ns;

            assert i_ready = '1'
                report
                    "FAIL: initial line filling ended before eight bytes were accepted."
                severity failure;


            ------------------------------------------------------------
            -- Insert one cycle with i_valid = 0 after byte 2.
            -- The initial-fill counter must not advance during this cycle.
            ------------------------------------------------------------

            if value = 2 then
                wait until rising_edge(clk);
                wait for 1 ns;

                assert i_ready = '1'
                    report
                        "FAIL: initial fill changed during an invalid input cycle."
                    severity failure;
            end if;

        end loop;


        ----------------------------------------------------------------
        -- Send the eighth and final byte.
        --
        -- Expected final write:
        --
        -- line_buffer(1, 3) = 7
        ----------------------------------------------------------------

        send_activation_byte(
            p_clk   => clk,
            p_valid => i_valid,
            p_ready => i_ready,
            p_data  => i_data,
            value   => C_INITIAL_FILL_SIZE - 1
        );

        wait for 1 ns;

        assert i_ready = '0'
            report
                "FAIL: i_ready remained high after the initial fill completed."
            severity failure;


        ----------------------------------------------------------------
        -- Allow the controller to observe initial_line_fill_done and
        -- transition to S_PRIME_K_LINE.
        ----------------------------------------------------------------

        wait until rising_edge(clk);
        wait until rising_edge(clk);
        wait for 1 ns;

        assert i_ready = '0'
            report
                "FAIL: i_ready unexpectedly returned high after initial fill."
            severity failure;


        ----------------------------------------------------------------
        -- Test passed
        ----------------------------------------------------------------

        report
            "PASS: S_INITIAL_LINE_FILL accepted exactly eight bytes."
            severity note;

        stop;
        wait;

    end process;

end architecture sim;