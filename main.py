import math

import pandas as pd


class hiddenMarkovModel:
    def __init__(self, observations, states, startingProbabilities, transitionProbabilities,
                 emissionProbabilities):
        self.observations=observations
        self.states=states
        self.startingProbabilities=startingProbabilities
        self.transitionProbabilities=transitionProbabilities
        self.emissionProbabilities=emissionProbabilities

def viterbi(hmm): # takes hidden markov model as parameter
    viterbiTable=pd.DataFrame(index=hmm.states, columns=range(len(hmm.observations)))
    for state in hmm.states:
        for observation in range(len(hmm.observations)):
            probability=0.0
            previousState="default"
            viterbiTable.loc[state, observation]=(probability, previousState)

    for state in hmm.states:
        viterbiTable.loc[state,0]=(hmm.startingProbabilities[state] * hmm.emissionProbabilities[state][hmm.observations[0]], None)

    for observation in range(1,len(hmm.observations)):
        for state in hmm.states:
            maxTransitionProbability=viterbiTable.loc[hmm.states[0],observation-1][0]*\
                                     hmm.transitionProbabilities[hmm.states[0]][state]*\
                                     hmm.emissionProbabilities[state][hmm.observations[observation]]
            bestPreviousState=hmm.states[0]
            for previousState in hmm.states[1:]:
                transitionProbability=viterbiTable.loc[previousState, observation-1][0]*\
                                      hmm.transitionProbabilities[previousState][state]*\
                                      hmm.emissionProbabilities[state][hmm.observations[observation]]
                if transitionProbability>maxTransitionProbability:
                    maxTransitionProbability=transitionProbability
                    bestPreviousState=previousState

            viterbiTable.loc[state, observation]=(maxTransitionProbability, bestPreviousState)

    # finds best path given final observation
    mostProbablePath=[]
    maxProbability=0.0
    bestState=None
    for state, info in viterbiTable.iloc[:, -1].items():
        if info[0]>maxProbability:
            maxProbability=info[0]
            bestState=state
    mostProbablePath.append(bestState)
    previous=bestState

    # fill out most probable path by traversing backwards
    for t in range(len(viterbiTable.columns)-2, -1, -1):
        mostProbablePath.insert(0, viterbiTable.loc[bestState, t+1][1])
        previous=viterbiTable.loc[previous, t+1][1]

    viterbiTable.columns=hmm.observations
    print("Viterbi table:\n", viterbiTable.to_string(), "\n")
    print("highest probability: " + str(maxProbability))
    print("most probable path: " + " ".join(mostProbablePath), "\n")

def forwardAlgorithm(hmm):
    forwardTable = pd.DataFrame(index=hmm.states, columns=range(len(hmm.observations)))
    for state in hmm.states:
        forwardTable.loc[state][0]=hmm.startingProbabilities[state] * hmm.emissionProbabilities[state][hmm.observations[0]]

    for observation in range(1, len(hmm.observations)):
        for state in hmm.states:
            jointProbability=sum(forwardTable.loc[nextState][observation-1]*hmm.transitionProbabilities[nextState][state]
                                 for nextState in hmm.states)
            jointProbability*=hmm.emissionProbabilities[state][hmm.observations[observation]]
            forwardTable.loc[state][observation]=jointProbability

    forwardTable.columns=hmm.observations

    return forwardTable

def backwardAlgorithm(hmm):
    backwardTable = pd.DataFrame(index=hmm.states, columns=range(len(hmm.observations)))
    backwardTable.iloc[:, -1]=1

    for observation in range(len(hmm.observations)-2, -2, -1):
        for state in hmm.states:
            backwardTable.loc[state][observation]=sum(hmm.transitionProbabilities[state][previousState] *
                                                      hmm.emissionProbabilities[previousState][hmm.observations[observation+1]] *
                                                      backwardTable.loc[previousState][observation+1] for previousState in hmm.states)

    backwardTable.columns=hmm.observations

    return backwardTable

def forwardBackwardAlgorithm(hmm):
    forwardTable=forwardAlgorithm(hmm)
    print("Forward table: \n", forwardTable.to_string(), "\n")
    backwardTable=backwardAlgorithm(hmm)
    print("Backward table: \n", backwardTable.to_string(), "\n")

    forwardProbability=forwardTable.iloc[:, -1].sum()
    backwardProbability=sum(hmm.startingProbabilities[state] * hmm.emissionProbabilities[state][hmm.observations[0]] *
                              backwardTable[hmm.observations[0]].iloc[:,0][state] for state in hmm.states)
    # i used iloc for backwardTable to get the first instance of hmm.observations[0], then i got the desired state from
    # the resulting array
    assert forwardProbability==backwardProbability

    posteriorProbabilities=pd.DataFrame(index=hmm.states, columns=hmm.observations)
    for observation in hmm.observations:
        for state in hmm.states:
            posteriorProbabilities.loc[state][observation]=forwardTable.loc[state][observation] * \
                                                           backwardTable.loc[state][observation] / forwardProbability
    print("Posterior probabilities:\n", posteriorProbabilities.to_string())

    return posteriorProbabilities

# creates a sample hmm and runs viterbi and forward backward algorithm
def dishonestCasino():
    observations=("H", "H", "H", "H", "H", "T", "T", "T", "T", "T")
    states=("F", "B")
    startingProbabilities={"F": 0.5, "B": 0.5}
    transitionProbabilities={
        "F": {"F": 0.9, "B": 0.1},
        "B": {"F": 0.1, "B": 0.9}
    }
    emissionProbabilities={
        "F": {"H": 0.5, "T": 0.5},
        "B": {"H": 0.75, "T": 0.25}
    }
    hmm = hiddenMarkovModel(observations=observations, states=states, startingProbabilities=startingProbabilities,
                                 transitionProbabilities=transitionProbabilities, emissionProbabilities=emissionProbabilities)
    viterbi(hmm)
    forwardBackwardAlgorithm(hmm)

dishonestCasino()