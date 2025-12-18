import optparse
import sys
import yaml
import math
import time
import gymnasium as gym
import gym_pusht
from VideoRecorder import VideoRecorder

from random import Random
from Population import *
from Evaluator import *


# Agent Config class
class AgentConfig:
    """
    Agent configuration class
    """
    # class variables
    sectionName='AgentConfig'
    options={'populationSize': (int,True),
             'generationCount': (int,True),
             'randomSeed': (int,True),
             'crossoverFraction': (float,True),
             'evaluator': (str,True),
             'nGenesCfg': (int,True),
             'geneRangeCfg': (int,True),
             'numTryPerMut': (int,True),
             'blockActionSize': (int,True),
             'selfCost' : (list,True),
             'selfDamage': (list,True),
             'EnhanceDamage': (list,True)}

    #constructor
    def __init__(self, inFileName):
        #read YAML config and get AgentConfig section
        infile=open(inFileName,'r')
        ymlcfg=yaml.safe_load(infile)
        infile.close()
        eccfg=ymlcfg.get(self.sectionName,None)
        if eccfg is None:
            raise Exception('Missing {} section in cfg file'.format(self.sectionName))

        #iterate over options
        for opt in self.options:
            if opt in eccfg:
                optval=eccfg[opt]

                #verify parameter type
                if type(optval) != self.options[opt][0]:
                    raise Exception('Parameter "{}" has wrong type'.format(opt))

                #create attributes on the fly
                setattr(self,opt,optval)
            else:
                if self.options[opt][1]:
                    raise Exception('Missing mandatory parameter "{}"'.format(opt))
                else:
                    setattr(self,opt,None)

    #string representation for class data
    def __str__(self):
        return str(yaml.dump(self.__dict__,default_flow_style=False))

#Print some useful stats to screen
def printStats(pop, gen):
    avgReward=0
    avgSeqSteps=0

    # objectives = [nSeqSteps, rewardEnd]
    nSeqSteps, maxReward = pop[0].objectives
    mutRate = pop[0].mutRate

    for ind in pop:
        avgSeqSteps += ind.objectives[0]   # steps
        avgReward   += ind.objectives[1]   # reward

        # find individual with max reward
        if ind.objectives[1] > maxReward:
            nSeqSteps, maxReward = ind.objectives
            mutRate = ind.mutRate
        # print(ind)
    print('Generation:',gen)
    print('Max Reward', maxReward)
    print('SeqSteps of Best', nSeqSteps)
    print('Avg Reward', avgReward/len(pop))
    print('Avg SeqSteps', avgSeqSteps/len(pop))
    print('MutRate', mutRate)
    print('')


#
# Helper function that allows us to init all cfg-related class
#  variables on our Pool worker processes
#
def initClassVars(cfg):
    Evaluator.selfCost=cfg.selfCost
    Evaluator.selfDamage=cfg.selfDamage
    Evaluator.EnhanceDamage=cfg.EnhanceDamage
    Evaluator.cfg = cfg # Make cfg available to the Evaluator class

    AgentIndividual.ObjFunc=Evaluator.ObjFunc
    AgentIndividual.nGenes=cfg.nGenesCfg
    AgentIndividual.geneRange=cfg.geneRangeCfg
    AgentIndividual.numTryPerMut=cfg.numTryPerMut
    AgentIndividual.blockActionSize=cfg.blockActionSize
    AgentIndividual.learningRate=1.0/math.sqrt(cfg.nGenesCfg)

    Population.individualType=AgentIndividual

#EV3_MO:
#
def EV3_MO(cfg):
    # start random number generators
    uniprng=Random()
    uniprng.seed(cfg.randomSeed)
    normprng=Random()
    normprng.seed(cfg.randomSeed+101)

    # set static params on classes
    # (probably not the most elegant approach, but let's keep things simple...)
    Individual.uniprng=uniprng
    Individual.normprng=normprng
    Population.uniprng=uniprng
    Population.crossoverFraction=cfg.crossoverFraction
    initClassVars(cfg)

    # create initial Population (random initialization)
    population=Population(cfg.populationSize)
    population.updateRanking()

    # print initial pop stats
    printStats(population, 0)
    # population.generatePlots(title=f'Generation 0')

    # evolution main loop
    for i in range(cfg.generationCount):
        # create initial offspring population by copying parent pop
        offspring=population.copy()

        # select mating pool

        #offspring.conductTournament()
        offspring.binaryTournament()

        #perform crossover
        offspring.crossover()

        #random mutation
        offspring.mutate()

        #Update objectives
        offspring.evaluateObjectives()

        #survivor selection: elitist truncation using parents+offspring
        population.combinePops(offspring)

        #Objectives have changed, so remember to update the ranking before truncation occurs.
        population.updateRanking()

        #population.truncateSelect(cfg.populationSize)
        population.MOTruncation(cfg.populationSize)

        #print population stats
        printStats(population,i+1)
        #print the objective space with its frontRank
        #population.generatePlots(title=f'Generation {i+1}')

    # Define weighting factors
    weight_reward = 1.0  # Example weighting for rewardEnd
    weight_nSeqSteps = 0 # Example weighting for nSeqSteps

    # After evolution, find the individual with the highest weighted score
    best_individual = None
    max_weighted_score = -float('inf')

    for ind in population:
        # objectives = [nSeqSteps, rewardEnd]
        nSeqSteps_val = ind.objectives[0]
        rewardEnd_val = ind.objectives[1]

        # Calculate weighted score
        current_weighted_score = (weight_reward * rewardEnd_val) - (weight_nSeqSteps * nSeqSteps_val)

        if current_weighted_score > max_weighted_score:
            max_weighted_score = current_weighted_score
            best_individual = ind

    if best_individual:
        print(f'\nBest individual (weighted score: {max_weighted_score:.4f}):')
        print(f'  Reward End: {best_individual.objectives[1]}')
        print(f'  Sequence Steps: {best_individual.objectives[0]}')
        print(f'  State (Action Sequence): {best_individual.state}')
        v = VideoRecorder(best_individual.state)
        v.record()
        return best_individual.state
    else:
        return None

#
# Main entry point
#
def main(argv=None):
    if argv is None:
        argv = sys.argv

    try:
        parser = optparse.OptionParser()
        parser.add_option("-i", "--input", action="store", dest="inputFileName", help="input filename", default="pushT/config_pushT.cfg")
        parser.add_option("-q", "--quiet", action="store_true", dest="quietMode", help="quiet mode", default=False)
        parser.add_option("-d", "--debug", action="store_true", dest="debugMode", help="debug mode", default=False)
        (options, args) = parser.parse_args(argv)

        # validate options
        if options.inputFileName is None:
            raise Exception("Must specify input file name using -i or --input option.")

        # Get Agent config params
        cfg=AgentConfig(options.inputFileName)

        #run EV3_MO
        start_time=time.asctime()
        final_best_state = EV3_MO(cfg)

        print('Start time: {}'.format(start_time))
        print('End time  : {}'.format(time.asctime()))

        if not options.quietMode:
            print('Agent Completed!')
            if final_best_state:
                print(f'Final best action sequence: {final_best_state}')
            else:
                print('No best action sequence found.')

    except Exception as info:
        from traceback import print_exc
        print_exc()


if __name__ == '__main__':
    main()  # Normal code line for use on desktop computer
    # main(argv=['-i', 'AgentConfig.cfg'])  # Code line to use for Colab