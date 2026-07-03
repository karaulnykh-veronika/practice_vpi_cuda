CXX = g++
CXXFLAGS = -I/opt/nvidia/vpi4/include -I/usr/include/opencv4
LDFLAGS = -L/opt/nvidia/vpi4/lib/x86_64-linux-gnu -lnvvpi -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

vpi_image: vpi_image.cpp
	$(CXX) $(CXXFLAGS) -o vpi_image vpi_image.cpp $(LDFLAGS)

clean:
	rm -f vpi_image