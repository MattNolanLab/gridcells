all: gaussian_arma

gaussian_arma:
	g++ gaussian_arma.cpp -o gaussian_arma -larmadillo -O2

run: gaussian_arma
	./gaussian_arma

timed: gaussian_arma
	time ./gaussian_arma
