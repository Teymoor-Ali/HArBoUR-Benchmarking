`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 
// Design Name: 
// Module Name: gaussian_blur
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 3x3 Gaussian Blur
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module gaussian_blur(
    input        i_clk,
    input [71:0] i_pixel_data,
    input        i_pixel_data_valid,
    output reg [7:0] o_convolved_data,
    output reg   o_convolved_data_valid
);

integer i; 
reg [7:0] kernel [8:0];
reg [15:0] multData[8:0];
reg [15:0] sumDataInt;
reg [15:0] sumData;
reg multDataValid;
reg sumDataValid;

initial begin
    // Gaussian kernel (3x3 example)
    kernel[0] = 1; kernel[1] = 2; kernel[2] = 1;
    kernel[3] = 2; kernel[4] = 4; kernel[5] = 2;
    kernel[6] = 1; kernel[7] = 2; kernel[8] = 1;
end

always @(posedge i_clk) begin
    if (i_pixel_data_valid) begin
        for (i = 0; i < 9; i = i + 1) begin
            multData[i] <= kernel[i] * i_pixel_data[i * 8 +: 8];
        end
        multDataValid <= i_pixel_data_valid;
    end
end

always @(*) begin
    sumDataInt = 0;
    for (i = 0; i < 9; i = i + 1) begin
        sumDataInt = sumDataInt + multData[i];
    end
end

always @(posedge i_clk) begin
    sumData <= sumDataInt;
    sumDataValid <= multDataValid;
end

always @(posedge i_clk) begin
    o_convolved_data <= sumData >> 4; // Normalizing by 16 (sum of kernel elements)
    o_convolved_data_valid <= sumDataValid;
end

endmodule
