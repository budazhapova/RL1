#include "Bandit.h"

// # of arms, # of runs, # of selections initialized in Bandit.h
int main(){
    Bandit gang;
    gang.reset();
    // pass chance of exploration epsilon as argument
    gang.epsilonGreedy(0.3);
    // pass learning rate alpha and initial valuation as argument
    gang.optimisticInitValues(0.1, 3.0);
    // pass learning rate alpha as argument
    gang.reinforcementComparison(0.7);
    // pass exploration rate c
    gang.upperConfidenceBound(1.0);
    gang.outputResults();
    return 0;
}