#include "Bandit.h"

int main(){
    Bandit mr_bandit;
    mr_bandit.reset();
    mr_bandit.epsilonGreedy(0.1);
    mr_bandit.outputResults();
    return 0;
}