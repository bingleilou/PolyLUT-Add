module polylut (input [47:0] M0, input clk, input rst, output[14:0] M6);
wire [47:0] M0w;
myreg #(.DataWidth(48)) layer0_reg (.data_in(M0), .clk(clk), .rst(rst), .data_out(M0w));
wire [511:0] M1;
layer0 layer0_inst (.M0(M0w), .M1(M1));
wire [191:0] M2;
adder0 adder0_inst (.M0(M1), .M1(M2));

wire [191:0] M2w;
myreg #(.DataWidth(192)) layer1_reg (.data_in(M2), .clk(clk), .rst(rst), .data_out(M2w));
wire [255:0] M3;
layer1 layer1_inst (.M0(M2w), .M1(M3));
wire [95:0] M4;
adder1 adder1_inst (.M0(M3), .M1(M4));

wire [95:0] M4w;
myreg #(.DataWidth(96)) layer2_reg (.data_in(M4), .clk(clk), .rst(rst), .data_out(M4w));
wire [39:0] M5;
layer2 layer2_inst (.M0(M4w), .M1(M5));
adder2 adder2_inst (.M0(M5), .M1(M6));

endmodule
