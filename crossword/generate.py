from multiprocessing.context import assert_spawning
import sys
from numpy import count_nonzero

from sklearn import neighbors

from crossword import *

import copy

class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for var, word in assignment.items():
            direction = var.direction
            for k in range(len(word)):
                i = var.i + (k if direction == Variable.DOWN else 0)
                j = var.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for var in self.crossword.variables:
            words = copy.deepcopy(self.domains[var])
            for word in words:
                if len(word) != var.length:
                    self.domains[var].remove(word)


    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """        
        if x != y and self.crossword.overlaps[x, y] is not None:
            xwords = copy.deepcopy(self.domains[x])
            for xword in xwords:
                revision = True
                ywords = copy.deepcopy(self.domains[y])
                for yword in ywords:
                    if xword[self.crossword.overlaps[x, y][0]] == yword[self.crossword.overlaps[x, y][1]]:
                        revision = False
                        break
                if revision == True:
                    self.domains[x].remove(xword)
        return revision
        

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        # If arcs is None, function should start with an initial queue of all of the arcs in the problem.
        if arcs is None:
            queue = []
            for xvar in self.crossword.variables:
                for yvar in self.crossword.neighbors(xvar):
                    # create a queue for all variables with their respective neighbors
                    queue.append((xvar, yvar))

        while len(queue) > 0:
            x, y = queue.pop()
            if self.revise(x, y) == True:
                if len(self.domains[x]) == 0:
                    return False
                for neighbor in self.crossword.neighbors(x):
                    if neighbor != y:
                        queue.append((neighbor, x))
        return True


    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        # check if each key(variable) has a non repeated value(word)
        # check if all variables are assigned a value
        if len(assignment) == len(list(assignment.values())) and len(assignment) == len(self.crossword.variables):
            return True
        else:
            return False


    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # check for duplicated assigned values
        if len(assignment.values()) != len(set(assignment.values())):
            return False

        # check if values have correct length
        for var, word in assignment.items():
            if len(word) != var.length:
                return False

        # check for conflicts between neighbouring variables
        for var, word in assignment.items():
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    (x, y) = self.crossword.overlaps[var, neighbor]
                    if word[x] != assignment[neighbor][y]:
                        return True
        return True


    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        # least-constraining values heuristic
        constrain = {}        
        for word in self.domains[var]:
            count = 0
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    continue
                for word_n in self.domains[neighbor]:
                    (x, y) = self.crossword.overlaps[var, neighbor]
                    if word[x] != word_n[y]:
                        count += 1
            constrain[word] = count
        constrain = sorted(constrain.items(), key=lambda x:x[1]) # constrain is now a list of words sorted by respective number of constrains
        
        ordered_domain = list()
        for word in constrain:
            # return only the sorted keys
            ordered_domain.append(word[0])
        
        return ordered_domain
        # alternatively, 
        # sorted_constrain = {k: v for k, v in sorted(constrain.items(), key=lambda x:x[1])}
        # return [*sorted_constrain]


    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # minimum remaining value heuristic and then the maximum degree heuristic
        unassigned = {}
        for var in self.crossword.variables:
            if var not in assignment:
                domain_count = len(self.domains[var])
                tot_neighbor = len(self.crossword.neighbors(var))
                unassigned[var] = (domain_count, tot_neighbor)
        
        # sort by domain_count(asc) first then tot_neighbor(desc)
        unassigned = sorted(unassigned.items(), key=lambda x:(x[1][0], -x[1][1]))
        return unassigned[0][0] # return the first variable in sorted list


    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """       
        if self.assignment_complete(assignment):
            return assignment
        else:
            # start with the variable that has minimum remaining value heuristic and maximum degree heuristic
            var = self.select_unassigned_variable(assignment)
            # start with word in domain that has least-constraining values heuristic
            for word in self.order_domain_values(var, assignment):
                assignment_temp = copy.deepcopy(assignment)
                assignment_temp[var] = word
                # check if the assignment is consistent with the constrains
                if self.consistent(assignment_temp):
                    result = self.backtrack(assignment_temp)
                    if result is not None:
                        return result
            return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
