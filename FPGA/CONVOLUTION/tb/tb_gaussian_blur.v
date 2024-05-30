`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 
// Design Name: 
// Module Name: tb_gaussian_blur
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

`define headerSize 1080
`define imageWidth 3840
`define imageHeight 2160
`define imageSize (`imageWidth * `imageHeight)

module tb_gaussian_blur();

    reg clk;
    reg reset;
    reg [7:0] imgData;
    integer file, file2, i;
    reg imgDataValid;
    integer sentSize;
    wire intr;
    wire [7:0] outDataGauss;
    wire outDataValid;
    integer receivedData = 0;
    time startTime, endTime;

    initial begin
        clk = 1'b0;
        forever begin
            #5 clk = ~clk;
        end
    end

    initial begin
        reset = 0;
        sentSize = 0;
        imgDataValid = 0;
        #100;
        reset = 1;
        #100;
        file = $fopen("test_image.bmp", "rb");
        file2 = $fopen("gaussian_filtered_test_image.bmp", "wb");
        for (i = 0; i < `headerSize; i = i + 1) begin
            $fscanf(file, "%c", imgData);
            $fwrite(file2, "%c", imgData);
        end
        
        for (i = 0; i < 4 * `imageWidth; i = i + 1) begin
            @(posedge clk);
            $fscanf(file, "%c", imgData);
            imgDataValid <= 1'b1;
        end
        sentSize = 4 * `imageWidth;
        @(posedge clk);
        imgDataValid <= 1'b0;
        startTime = $time;
        while (sentSize < `imageSize) begin
            @(posedge intr);
            for (i = 0; i < `imageWidth; i = i + 1) begin
                @(posedge clk);
                $fscanf(file, "%c", imgData);
                imgDataValid <= 1'b1;    
            end
            @(posedge clk);
            imgDataValid <= 1'b0;
            sentSize = sentSize + `imageWidth;
        end
        @(posedge clk);
        imgDataValid <= 1'b0;
        @(posedge intr);
        for (i = 0; i < `imageWidth; i = i + 1) begin
            @(posedge clk);
            imgData <= 0;
            imgDataValid <= 1'b1;    
        end
        @(posedge clk);
        imgDataValid <= 1'b0;
        @(posedge intr);
        for (i = 0; i < `imageWidth; i = i + 1) begin
            @(posedge clk);
            imgData <= 0;
            imgDataValid <= 1'b1;    
        end
        @(posedge clk);
        imgDataValid <= 1'b0;
        $fclose(file);
    end

    always @(posedge clk) begin
        if (outDataValid) begin
            $fwrite(file2, "%c", outDataGauss);
            receivedData = receivedData + 1;
        end 
        if (receivedData == `imageSize) begin
            endTime = $time;
            $fclose(file2);
            $display("Gaussian Blur Processing Time: %0t ns", endTime - startTime);
            $stop;
        end
    end

    // Instantiate the image processing top module
    imageProcessTop dut(
        .axi_clk(clk),
        .axi_reset_n(reset),
        // Slave interface
        .i_data_valid(imgDataValid),
        .i_data(imgData),
        .o_data_ready(),
        // Master interface
        .o_data_valid(outDataValid),
        .o_data(outDataGauss), // Connect to Gaussian Blur output for now
        .i_data_ready(1'b1),
        // Interrupt
        .o_intr(intr)
    );

    // Instantiate the Gaussian Blur
    gaussian_blur gaussian_blur_inst(
        .i_clk(clk),
        .i_pixel_data(dut.pixel_data),
        .i_pixel_data_valid(dut.pixel_data
