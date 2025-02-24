"""
Tabu Search Class
"""
from log.logger import Logger


class TabuSearch:
    def __init__(self, initialSolution, solutionEvaluator, neighborOperator, #aspirationCriteria,
                 acceptableScoreThreshold, tabuTenure):
        self.currSolution = initialSolution
        self.bestSolution = initialSolution
        self.evaluate = solutionEvaluator
        # self.aspirationCriteria = aspirationCriteria
        self.neighborOperator = neighborOperator
        self.acceptableScoreThreshold = acceptableScoreThreshold
        self.tabuTenure = tabuTenure
        self.terminationCount = 0

    def is_termination_criteria_met(self):
        # can add more termination criteria
        return self.evaluate(self.bestSolution) < self.acceptableScoreThreshold \
               or self.neighborOperator(self.currSolution) == 0 \
               or self.terminationCount > 100

    def aspirationCriteria(self, neighbour, tabuSolutions):
        """check if neighbour is in TabuList"""
        if neighbour in tabuSolutions:
            return False
        else:
            return True

    def run(self):
        tabuList = {}
        curr_eval_metric = None

        while not self.is_termination_criteria_met():
            # add to termination count
            self.terminationCount += 1

            # get all of the neighbors
            neighbors = self.neighborOperator(self.currSolution)

            # find all tabuSolutions other than those
            # that fit the aspiration criteria
            tabuSolutions = tabuList.keys()

            # find all neighbors that are not part of the Tabu list
            neighbors = list(filter(lambda n: self.aspirationCriteria(n, tabuSolutions), neighbors))

            # break out of while loop if no neighbours left after filter
            if len(neighbors) == 0:
                break

            # pick the best neighbor solution
            newSolution = sorted(neighbors, key=lambda n: self.evaluate(n))[0]

            # get the cost between the two solutions
            cost = self.evaluate(self.bestSolution) - self.evaluate(newSolution)

            # if the new solution is better,
            # update the best solution with the new solution
            if cost >= 0:
                self.bestSolution = newSolution
            # update the current solution with the new solution
            self.currSolution = newSolution

            # decrement the Tabu Tenure of all tabu list solutions
            sol_to_del = []
            for sol in tabuList:
                tabuList[sol] -= 1
                if tabuList[sol] == 0:
                    sol_to_del.append(sol)

            # delete empty tabuList
            for sol in sol_to_del:
                del tabuList[sol]

            # add new solution to the Tabu list
            tabuList[newSolution] = self.tabuTenure

            curr_eval_metric = self.evaluate(self.currSolution)

        # return best solution found
        return self.bestSolution, curr_eval_metric