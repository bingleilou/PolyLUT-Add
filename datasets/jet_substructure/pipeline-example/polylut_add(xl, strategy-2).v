module polylut_add (input [111:0] M0, input clk, input rst, output[24:0] M10);
wire [111:0] M0w;
myreg #(.DataWidth(112)) layer0_reg (.data_in(M0), .clk(clk), .rst(rst), .data_out(M0w));
wire [1535:0] M1;
layer0 layer0_inst (.M0(M0w), .M1(M1));
wire [639:0] M2;
adder0 adder0_inst (.M0(M1), .M1(M2));

wire [639:0] M2w;
myreg #(.DataWidth(640)) layer1_reg (.data_in(M2), .clk(clk), .rst(rst), .data_out(M2w));
wire [767:0] M3;
layer1 layer1_inst (.M0(M2w), .M1(M3));
wire [319:0] M4;
adder1 adder1_inst (.M0(M3), .M1(M4));

wire [319:0] M4w;
myreg #(.DataWidth(320)) layer2_reg (.data_in(M4), .clk(clk), .rst(rst), .data_out(M4w));
wire [767:0] M5;
layer2 layer2_inst (.M0(M4w), .M1(M5));
wire [319:0] M6;
adder2 adder2_inst (.M0(M5), .M1(M6));

wire [319:0] M6w;
myreg #(.DataWidth(320)) layer3_reg (.data_in(M6), .clk(clk), .rst(rst), .data_out(M6w));
wire [767:0] M7;
layer3 layer3_inst (.M0(M6w), .M1(M7));
wire [319:0] M8;
adder3 adder3_inst (.M0(M7), .M1(M8));

wire [319:0] M8w;
myreg #(.DataWidth(320)) layer4_reg (.data_in(M8), .clk(clk), .rst(rst), .data_out(M8w));
wire [59:0] M9;
layer4 layer4_inst (.M0(M8w), .M1(M9));
adder4 adder4_inst (.M0(M9), .M1(M10));

endmodule
