# Canny Edge Detector
## FESB

### Contributors
* Dea Čelan
* Sara Ćurak
* Luka Jelavić Šako
* Dino Zoričić

### Setup

This project is made in Google Colab.<br/>
To set it up, open the Jupyter Notebook file in Google colab

* Important: Go to runtime -> Change runtime type -> select 'T4 GPU' -> Save<br/>You will be prompted to restart your session. (This is needed to run cude/gpu related functions such as memory allocation)<br/>

Zip the rest of the files into <strong>_canny.zip_</strong> file and upload it to Google colab

* Important: it has to be <strong>zip</strong> file. Rar or 7z files are not supported by default.

Follow the instructions on colab to unzip, compile and run the app.<br/>

### Jupyter Notebook

Notebook consists of several segments
* Unzip canny.zip
* Compile libs and main file (cu and cpp)
* Run main
* Cleanup and zip

#### Unzip
Unzip is used to quickly transfer multiple files into Google colab. Instead of moving one by one file and creating directories, you can move the zip file and unzip it.

#### Compile
There are several files that need to be compiled to run the program
* External libs (for saving image in cpp and getting metadata - height, width...) 
* Internal libs (image handler - common functions for converting image to array and vice versa)
* Canny edge detector libs (cpu, gpu and gpu with shared memory - All functions needed to run Canny edge detector written specifically for each case)
* Main.cu and input.jpg (main project with image used for testing - more files or different named files are currently not supported)

#### Run
Running the main file runs the cpu, gpu and gpu with shared memory functions sequentially, with timer starting and stopping before and after each function.<br/>
With this approach, we tested all three cases in one go, with final times displayed as log output bellow Jupyter Notebook cells.

#### Cleanup
Cleanup is used to remove the compiled files (*.o *.out) and output images (*.jpg).<br/>
With this we can quickly retest compiling and execution of the program.

This also gives us the ability to zip the project after developing to easily download it as one file instead of manually downloading multiple files.