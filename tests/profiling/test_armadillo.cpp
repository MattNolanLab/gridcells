#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;


void changeElement(mat& M, double fill)
{
    M(0, 0) = fill;
}


int main(void)
{

    mat M = randu(5, 5);
    mat K = M;

    M.print("M: ");
    K.print("K: ");
    cout << endl << endl;

    K(0, 0) = 50.;

    M.print("M: ");
    K.print("K: ");
    cout << endl << endl;

    changeElement(M, 100);

    M.print("M: ");
    K.print("K: ");

    return 0;
}
