`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 
// Design Name: 
// Module Name: box_filter
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 3x3 Box Filter
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module box_filter(
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
    for (i = 0; i < 9; i = i + 1) begin
        kernel[i] = 1; // Box filter kernel (all ones)
    end
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
    o_convolved_data <= sumData / 9; // Normalizing by 9 (number of pixels in 3x3 window)
    o_convolved_data_valid <= sumDataValid;
end

endmodule
