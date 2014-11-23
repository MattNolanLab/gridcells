%module spikes

%include <std_pair.i>
%include <std_string.i>

%{
#define SWIG_FILE_WITH_INIT
#include "spikes.hpp"
%}

%include "armanpy.i"

%template() std::pair<arma::mat*, arma::vec>;
%template() std::pair<int, std::string *>;

%include "spikes.hpp"
