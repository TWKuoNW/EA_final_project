import math

#Base class for all individual types
#
class Individual:
    """
    Individual
    """
    minMutRate=1e-100
    maxMutRate=1
    learningRate=None
    uniprng=None
    normprng=None
    ObjFunc=None

    def __init__(self):
        self.objectives=self.__class__.ObjFunc(self.state)
        self.mutRate=self.uniprng.uniform(0.9,0.1) #use "normalized" sigma
        self.numObj=len(self.objectives)
        self.frontRank=None
        self.crowdDist=None

    def mutateMutRate(self):
        self.mutRate=self.mutRate*math.exp(self.learningRate*self.normprng.normalvariate(0,1))
        if self.mutRate < self.minMutRate:
            self.mutRate=self.minMutRate
        if self.mutRate > self.maxMutRate:
            self.mutRate=self.maxMutRate

    def evaluateObjectives(self):
        if self.objectives == None:
            self.objectives=self.__class__.ObjFunc(self.state)

    def dominates(self,other):
        dominatesCount=0
        equalsCount=0
        inferiorCount=0

        i=0
        while i < self.numObj:
            ### minimize mana cost
            if i==0:
                if self.objectives[i] < other.objectives[i]: dominatesCount+=1
                elif self.objectives[i] > other.objectives[i]: inferiorCount+=1
                else: equalsCount+=1
                i+=1

            ### maximize spell damage
            else:
                if self.objectives[i] > other.objectives[i]: dominatesCount+=1
                elif self.objectives[i] < other.objectives[i]: inferiorCount+=1
                else: equalsCount+=1
                i+=1

        if equalsCount == self.numObj: return 0 # the two are non-dominating
        elif dominatesCount+equalsCount == self.numObj: return 1 # self dominates other
        elif inferiorCount+equalsCount == self.numObj: return -1 # other dominates self
        else: return 0 # the two are non-dominating

    def compareRankAndCrowding(self, other):
        if self is other: return 0

        if self.frontRank < other.frontRank: return 1
        elif self.frontRank > other.frontRank: return -1
        else:
            if self.crowdDist > other.crowdDist: return 1
            elif self.crowdDist < other.crowdDist: return -1
            else: return 0

    def distance(self, other, normalizationVec=[None]):
        """
        Compute distance between self & other in objective space
        """
        # check if self vs self
        if self is other:
            return 0.0

        #set default normalization to 1.0, if not specified
        if normalizationVec[0] == None:
            normalizationVec=[1.0]*self.numObj

        # compute normalized Euclidian distance
        distance=0
        i=0
        while i < self.numObj:
            tmp=(self.objectives[i]-other.objectives[i])/normalizationVec[i]
            distance+=(tmp*tmp)
            i+=1

        distance=math.sqrt(distance)

        return distance

class AgentIndividual(Individual):
    """
    AgentIndividual
    """
    nGenes=None
    geneRange=None
    numTryPerMut = None
    blockActionSize = None

    def __init__(self):
        self.state = []
        init_genes = 2 * self.__class__.blockActionSize
        for _ in range(init_genes):
            self.state.append(self.uniprng.uniform(0.0, float(self.geneRange)))
        super().__init__()

    def crossover(self, other):
        #perform crossover "in-place"
        for i in range(self.nGenes):
            if self.uniprng.random() < 0.5:
                tmp=self.state[i]
                self.state[i]=other.state[i]
                other.state[i]=tmp

        self.objectives=None
        other.objectives=None

    def mutate(self):

        # Adapt mutation rate
        self.mutateMutRate()

        # Use the adapted mutation rate for existing gene mutations
        mutation_of_existing_genes_rate = self.mutRate

        if self.uniprng.random() < mutation_of_existing_genes_rate:
          # Mutating the existing genes
          n_mutable_genes = len(self.state) - 2

          if n_mutable_genes > 0:
              # Define the standard deviation for the Gaussian mutation
              # (e.g., 5% of the geneRange, can be adjusted)
              mutation_strength_sigma = float(self.geneRange) * 0.05

              for i in range(2, len(self.state)): # Iterate from the third gene (index 2)
                  # Calculate the index relative to the mutable range (0 to n_mutable_genes - 1)
                  index_in_mutable_range = i - 2

                  # Calculate mutation probability for this gene
                  if n_mutable_genes == 1:
                      # If only one mutable gene, it's at the beginning, so 5% probability
                      mutation_prob = 0.05
                  else:
                      # Linear decrease from 0.05 at start (index_in_mutable_range=0) to 0 at end
                      mutation_prob = 0.05 * (n_mutable_genes - 1 - index_in_mutable_range) / (n_mutable_genes - 1)

                  if self.uniprng.random() < mutation_prob:
                      # Mutate by adding a value from a Gaussian distribution centered at the current gene value
                      current_gene_value = self.state[i]
                      new_gene_value = self.normprng.normalvariate(current_gene_value, mutation_strength_sigma)

                      # Ensure the new value stays within the geneRange
                      self.state[i] = max(0.0, min(float(self.geneRange), new_gene_value))

        else:
          # Otherwise only additional genes are added

          # If already at maximum allowed length, do nothing
          if len(self.state) >= self.__class__.nGenes:
              return

          # Determine the maximum number of *actions* that can be added in one mutation step
          max_actions_per_mutation = self.__class__.blockActionSize

          # Randomly choose how many *actions* to add (between 0 and max_actions_per_mutation)
          num_actions_to_add = self.uniprng.randint(0, max_actions_per_mutation)

          # Convert actions to genes (each action uses 2 genes)
          genes_to_add_randomly = num_actions_to_add * 2

          # Avoid exceeding nGenes (the absolute maximum length for the state)
          max_addable_genes = self.__class__.nGenes - len(self.state)
          actual_genes_to_add = min(genes_to_add_randomly, max_addable_genes)

          # Append genes (in pairs: [a, b])
          # Ensure we only add an even number of genes since actions are pairs
          for _ in range(actual_genes_to_add // 2):
              a = self.uniprng.uniform(0.0, float(self.geneRange))
              b = self.uniprng.uniform(0.0, float(self.geneRange))
              self.state.extend([a, b])

        # Let the outer EA re-evaluate this individual later
        self.objectives = None

    def __str__(self):
        return (
            str(self.state) + '\t' +
            '%0.8e' % self.mutRate + '\t' +
            'nSeqSteps: ' + str(self.objectives[0]) + '\t' +
            'rewardEnd: ' + str(self.objectives[1]) + '\t' +
            'crowdDist: ' + '%0.8e' % self.crowdDist + ' ' +
            'frontRank: ' + str(self.frontRank)
        )
