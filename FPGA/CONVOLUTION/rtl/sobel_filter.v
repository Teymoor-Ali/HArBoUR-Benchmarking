`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 
// Design Name: 
// Module Name: sobel_filter
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 3x3 Sobel Filter
// 
// Dependencies: 
// 
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module sobel_filter(
    input        i_clk,
    input [71:0] i_pixel_data,
    input        i_pixel_data_valid,
    output reg [7:0] o_convolved_data,
    output reg   o_convolved_data_valid
);

integer i; 
reg signed [7:0] Gx [8:0];
reg signed [7:0] Gy [8:0];
reg signed [15:0] multDataX[8:0];
reg signed [15:0] multDataY[8:0];
reg [15:0] sumDataXInt;
reg [15:0] sumDataYInt;
reg [15:0] sumDataX;
reg [15:0] sumDataY;
reg multDataValid;
reg sumDataValid;

initial begin
    // Sobel kernel (Gx and Gy)
    Gx[0] = -1; Gx[1] = 0; Gx[2] = 1;
    Gx[3] = -2; Gx[4] = 0; Gx[5] = 2;
    Gx[6] = -1; Gx[7] = 0; Gx[8] = 1;

    Gy[0] = -1; Gy[1] = -2; Gy[2] = -1;
    Gy[3] =  0; Gy[4] =  0; Gy[5] =  0;
    Gy[6] =  1; Gy[7] =  2; Gy[8] =  1;
end

always @(posedge i_clk) begin
    if (i_pixel_data_valid) begin
        for (i = 0; i < 9; i = i + 1) begin
            multDataX[i] <= Gx[i] * i_pixel_data[i * 8 +: 8];
            multDataY[i] <= Gy[i] * i_pixel_data[i * 8 +: 8];
        end
        multDataValid <= i_pixel_data_valid;
    end
end

always @(*) begin
    sumDataXInt = 0;
    sumDataYInt = 0;
    for (i = 0; i < 9; i = i + 1) begin
        sumDataXInt = sumDataXInt + multDataX[i];
        sumDataYInt = sumDataYInt + multDataY[i];
    end
end

always @(posedge i_clk) begin
    sumDataX <= sumDataXInt;
    sumDataY <= sumDataYInt;
    sumDataValid <= multDataValid;
end

always @(posedge i_clk) begin
    if (sumDataValid) begin
        o_convolved_data <= abs(sumDataX) + abs(sumDataY); // Edge magnitude
        o_convolved_data_valid <= sumDataValid;
    end else begin
        o_convolved_data_valid <= 0;
    end
end

// Function to calculate absolute value
function [7:0] abs(input signed [15:0] val);
    begin
        if (val < 0)
            abs = -val;
        else
            abs = val;
    end
endfunction

endmodule
