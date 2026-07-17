library ieee;
use ieee.std_logic_1164.all;

library std;
use std.env.all;

use work.conv_tb_pkg.all;

entity tb_conv_layer_initial_line_fill is
end entity tb_conv_layer_initial_line_fill;


architecture sim of tb_conv_layer_initial_line_fill is

    constant C_CLK_PERIOD : time := 10 ns;

    constant C_C_IN   : positive := 1;
    constant C_W_IN   : positive := 4;
    constant C_KERNEL : positive := 3;

    constant C_INITIAL_FILL_SIZE : positive :=
        (C_KERNEL - 1) * C_W_IN * C_C_IN;

    signal clk   : std_logic := '0';
    signal rst_n : std_logic := '0';

    signal i_valid : std_logic := '0';
    signal i_ready : std_logic;
    signal i_data  : std_logic_vector(7 downto 0) :=
        (others => '0');

begin

    clk <= not clk after C_CLK_PERIOD / 2;

    dut : entity work.conv_layer
        generic map (
            G_C_IN   => C_C_IN,
            G_W_IN   => C_W_IN,
            G_C_PAR  => 2,
            G_KERNEL => C_KERNEL
        )
        port map (
            clk            => clk,
            rst_n          => rst_n,

            i_valid        => i_valid,
            i_ready        => i_ready,
            i_data         => i_data,

            i_weight_valid => '0',
            o_weight_ready => open,
            i_weight_data  => (others => '0'),
            o_acc_valid => open,
            o_acc_data  => open
        );


    stimulus_process : process
    begin
        reset_dut(
            p_clk   => clk,
            p_rst_n => rst_n,
            p_valid => i_valid,
            p_data  => i_data
        );

        wait until rising_edge(clk);
        wait for 1 ns;

        assert i_ready = '1'
            report "FAIL: initial line filling did not start."
            severity failure;

        send_activation_range(
            p_clk         => clk,
            p_valid       => i_valid,
            p_ready       => i_ready,
            p_data        => i_data,
            p_first_value => 0,
            p_count       => C_INITIAL_FILL_SIZE
        );

        wait for 1 ns;

        assert i_ready = '0'
            report
                "FAIL: initial line filling did not stop after the expected number of bytes."
            severity failure;

        report
            "PASS: S_INITIAL_LINE_FILL accepted exactly eight bytes."
            severity note;

        stop;
        wait;
    end process;

end architecture sim;