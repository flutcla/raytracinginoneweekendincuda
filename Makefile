out.jpg: out.ppm
	rm -f out.jpg
	ppmtojpeg out.ppm > out.jpg


out.ppm: rt
	rm -f out.ppm
	time ./rt > out.ppm

a.out: main.cpp
	g++ main.cpp -o rt

clean:
	rm -f rt out.ppm out.jpg