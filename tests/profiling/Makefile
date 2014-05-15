all: gaussian_arma fields

gaussian_arma: gaussian_arma.cpp
	g++ gaussian_arma.cpp -o gaussian_arma -larmadillo -O2

run: gaussian_arma
	./gaussian_arma

timed: gaussian_arma
	time ./gaussian_arma


test_armadillo: test_armadillo.cpp
	g++ $< -o test_armadillo -larmadillo -O2


fields: fields.cpp
	g++ $< -o fields -larmadillo -O3 -march=native -DARMA_NO_DEBUG


clean:
	rm -rf gaussian_arma test_armadillo fields
