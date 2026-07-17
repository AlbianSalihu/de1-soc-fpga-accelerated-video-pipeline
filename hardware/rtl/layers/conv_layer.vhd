library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity conv_layer is
    generic (
        G_C_IN : positive;
        G_W_IN : positive;
        G_C_PAR : positive;
        G_KERNEL : positive
    );
    port (
        clk : in std_logic;
        rst_n : in std_logic;

        i_valid : in std_logic;
        i_ready : out std_logic;
        i_data : in std_logic_vector(7 downto 0);

        i_weight_valid : in std_logic; 
        o_weight_ready : out std_logic;
        i_weight_data : in std_logic_vector(7 downto 0)
    );
end entity conv_layer;

architecture rtl of conv_layer is

    type conv_states is (
        S_IDLE,
        S_INITIAL_LINE_FILL,
        S_PRIME_K_LINE,
        S_CALC_AND_SLIDING_WINDOW,
        S_STREAM_LINE_FILLING,
        S_LINE_ROTATION
    );

    signal state : conv_states := S_IDLE;

    constant C_LINE_SIZE : positive := G_W_IN * G_C_IN;
    constant C_INITIAL_FILL_SIZE : positive := (G_KERNEL - 1) * C_LINE_SIZE;
    constant C_PRIME_FILL_SIZE : positive := G_KERNEL * G_C_IN;
    constant C_WEIGHT_FILL_SIZE : positive := G_C_PAR * G_KERNEL * G_KERNEL;

    type t_weight_buffer is array(
        0 to C_WEIGHT_FILL_SIZE - 1
    ) of signed(7 downto 0);

    type t_line_buffer is array (
        0 to G_KERNEL,
        0 to C_LINE_SIZE - 1
    ) of std_logic_vector(7 downto 0);

    signal weight_buffer : t_weight_buffer;

    signal weight_fill_active : std_logic := '0';
    signal weight_group_ready : std_logic := '0';
    signal weight_accepted : std_logic;
    signal weight_fill_count : natural range 0 to C_WEIGHT_FILL_SIZE - 1 := 0; 

    signal line_buffer : t_line_buffer;

    signal start_line_fill : std_logic;
    signal start_weight_fill : std_logic;
    signal start_prime_k_line : std_logic;

    signal activation_accepted : std_logic;

    signal initial_fill_active : std_logic := '0';
    signal initial_line_fill_done : std_logic := '0';

    signal initial_fill_count : natural range 0 to C_INITIAL_FILL_SIZE - 1 := 0;

    signal prime_k_line_active : std_logic := '0';
    signal first_window_ready : std_logic := '0';

    signal prime_fill_count : natural range 0 to C_PRIME_FILL_SIZE - 1 := 0;

    signal line_buffer_we : std_logic;

    signal line_buffer_wr_row : natural range 0 to G_KERNEL := 0;

    signal line_buffer_wr_addr : natural range 0 to C_LINE_SIZE - 1 := 0;

begin

    start_line_fill   <= '1' when state = S_IDLE else '0';
    start_weight_fill <= '1' when state = S_IDLE else '0';

    start_prime_k_line <= initial_line_fill_done;

    i_ready <= initial_fill_active or prime_k_line_active;

    activation_accepted <= i_valid and (initial_fill_active or prime_k_line_active);

    line_buffer_we <= activation_accepted;

    line_buffer_wr_row <= initial_fill_count / C_LINE_SIZE when initial_fill_active = '1' else G_KERNEL - 1;

    line_buffer_wr_addr <= initial_fill_count mod C_LINE_SIZE when initial_fill_active = '1' else prime_fill_count;

    o_weight_ready <= weight_fill_active;
    weight_accepted <= i_weight_valid and weight_fill_active;

    controller_process : process(clk)
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                state <= S_IDLE;

            else
                case state is

                    when S_IDLE =>
                        state <= S_INITIAL_LINE_FILL;

                    when S_INITIAL_LINE_FILL =>
                        if initial_line_fill_done = '1' then
                            state <= S_PRIME_K_LINE;
                        end if;

                    when S_PRIME_K_LINE =>
                        if first_window_ready = '1' and weight_group_ready = '1' then 
                            state <= S_CALC_AND_SLIDING_WINDOW;
                        end if;

                    when S_CALC_AND_SLIDING_WINDOW =>
                        null;

                    when S_STREAM_LINE_FILLING =>
                        null;

                    when S_LINE_ROTATION =>
                        null;

                end case;
            end if;
        end if;
    end process;


    initial_line_fill_process : process(clk)
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                initial_fill_active <= '0';
                initial_line_fill_done <= '0';
                initial_fill_count <= 0;

            else
                initial_line_fill_done <= '0';
                if start_line_fill = '1' then
                    initial_fill_active <= '1';
                    initial_fill_count <= 0;

                elsif initial_fill_active = '1' then
                    if activation_accepted = '1' then
                        if initial_fill_count = C_INITIAL_FILL_SIZE - 1 then
                            initial_fill_active <= '0';
                            initial_line_fill_done <= '1';

                        else
                            initial_fill_count <= initial_fill_count + 1;
                        end if;

                    end if;
                end if;
            end if;
        end if;
    end process;


    prime_k_line_process : process(clk)
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                prime_k_line_active <= '0';
                first_window_ready <= '0';
                prime_fill_count <= 0;

            else
                if start_prime_k_line = '1' then
                    prime_k_line_active <= '1';
                    first_window_ready <= '0';
                    prime_fill_count <= 0;

                elsif prime_k_line_active = '1' then
                    if activation_accepted = '1' then

                        if prime_fill_count = C_PRIME_FILL_SIZE - 1 then
                            prime_k_line_active <= '0';
                            first_window_ready <= '1';

                        else
                            prime_fill_count <= prime_fill_count + 1;
                        end if;

                    end if;
                end if;
            end if;
        end if;
    end process;


    line_buffer_write_process : process(clk)
    begin
        if rising_edge(clk) then
            if line_buffer_we = '1' then
                line_buffer(line_buffer_wr_row, line_buffer_wr_addr) <= i_data;
            end if;
        end if;
    end process;


    weight_filling_process : process(clk)
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                weight_fill_active <= '0';
                weight_group_ready <= '0';
                weight_fill_count <= 0;

            else
                if start_weight_fill = '1' then
                    weight_fill_active <= '1';
                    weight_group_ready <= '0';
                    weight_fill_count <= 0;

                elsif weight_fill_active = '1' then
                    if weight_accepted = '1' then
                        weight_buffer(weight_fill_count) <= signed(i_weight_data);

                        if weight_fill_count = C_WEIGHT_FILL_SIZE - 1 then
                            weight_fill_active <= '0';
                            weight_group_ready <= '1'; 
                        else
                            weight_fill_count <= weight_fill_count + 1;
                        end if;
                    end if;
                end if;
            end if;
        end if;
    end process;

end architecture rtl;