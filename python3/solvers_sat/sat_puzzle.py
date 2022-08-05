from itertools import chain
import pycosat
from typing import Any, Dict, Generator, Iterator, List, Set, Tuple

class SatPuzzleBase:
    '''
    Base class for a puzzle to SAT reducer. Must override the following:
    - toCnf: convert a puzzle instance to a CNF SAT instance
    - toSol: convert the solution to the SAT problem back to the puzzle solution
    The other functions extend the functionality of these and should not need to
    be overridden by subclasses.
    '''
    def toCnf(self) -> List[List[int]]:
        '''
        Convert puzzle to a CNF logic expression. The return value is a list of
        clauses similar to the DIMACS CNF format. Each clause is a list of
        nonzero integers (+n for variable n and -n for the negation of variable
        n). The value 0 should not be included, it is simply the clause
        terminator for the DIMACS CNF format and is not necessary for this list
        of lists representation.
        '''
        assert 0, 'not implemented'
        return []
    def toSol(self, satSol: List[int]) -> Any:
        '''
        Convert a solution to the CNF SAT instance into a puzzle solution. The
        type returned should be determined by the subclass.
        '''
        assert 0, 'not implemented'
    def cnfSolve(self) -> List[int]:
        '''
        Solves the CNF problem returned by self.toCnf(). Currently, pycosat
        (Python3 package for PicoSAT) is used. The return value is a list of
        nonzero integers (+n for n true and -n for n false), or the empty list
        if no solution is found.
        '''
        result = pycosat.solve(self.toCnf())
        assert result != 'UNKNOWN'
        return [] if result == 'UNSAT' else result
    def cnfSolveAll(self) -> Iterator[List[int]]:
        '''
        Finds all solutions to the CNF problem, returning an iterator of them.
        '''
        return pycosat.itersolve(self.toCnf())
    def solve(self) -> Any:
        '''
        Finds a solution to the logic puzzle. Currently, an exception should
        occur if no solution is found.
        '''
        return self.toSol(self.cnfSolve())
    def solveAll(self) -> Iterator[Any]:
        '''
        Returns an iterator of all solutions to the logic puzzle.
        '''
        return map(self.toSol,self.cnfSolveAll())

class SatPuzzleSudokuGeneral(SatPuzzleBase):
    '''
    A generalization of Sudoku, representing a puzzle as C > 0 cells (numbered
    0, 1, ..., C-1), a parameter N > 0, and sets of N cells (areas) which are
    constrained to contain the numbers 1, 2, ..., N, each exactly once.
    '''
    def __init__(self, cells: int, nums: int, areas: List[List[int]], givens: List[int]):
        '''
        cells = number of cells, numbered starting from 0
        nums = symbols in puzzle, represented as 1, 2, .., N
        areas = list of cells constrained to contain 1..N, each of size N
        givens = list of values given, 0 for no given value
        '''
        assert cells > 0
        assert nums > 0
        assert all(len(set(a)) == len(a) == nums for a in areas) # correct size
        assert all(all(0 <= c < cells for c in a) for a in areas) # valid index
        assert all(0 <= n <= nums for n in givens)
        self.cells = cells
        self.nums = nums
        self.areas = [a[:] for a in areas]
        self.givens = givens[:]
    def toCnf(self) -> List[List[int]]:
        '''
        variables: x(c,n) (0 <= c < cells, 1 <= n <= N)
        constraints:
        - each cell (c) has >= 1 value
          - x(c,1) or x(c,2) or ... or x(c,n)
        - each cell (c) has <= 1 value
          - express as for any pair, either number is not assigned to cell c
          - not x(c,a) or not x(c,b) (for 1 <= a < b <= N)
        - each area (cells a1,a2,..,aN) contains each number
          - x(a1,i) or x(a2,i) or ... or x(an,i) (for 1 <= i <= N)
        - each area (cells a1,a2,..,aN)) does not have a duplicate (redundant,
          necessary for efficiency) (for any 2 cells, either does not have n)
          - not x(a_i,n) or not x(a_j,n) (for each 1 <= n <= N and 1 <= i < j <= N)
        - use the given clues
          - x(c,i) (for each cell c with a given value i)
        '''
        result: List[List[int]] = []
        x = lambda c,n : 1 + c*self.nums + (n-1)
        for c in range(self.cells): # each cell has a value
            result.append([x(c,n) for n in range(1,self.nums+1)])
        for c in range(self.cells): # each cell has at most 1 value (for any 2 distinct values, one is not assigned to that cell)
            for n1 in range(1,self.nums+1):
                for n2 in range(n1+1,self.nums+1):
                    result.append([-x(c,n1),-x(c,n2)])
        for area in self.areas: # for each area
            for n in range(1,self.nums+1): # has each number
                result.append([x(c,n) for c in area])
                # add redundant clauses for efficiency
                for i,c1 in enumerate(area): # for any 2 cells, one does not have n (no duplicated numbers in an area)
                    for c2 in area[i+1:]:
                        result.append([-x(c1,n),-x(c2,n)])
        for c,n in enumerate(self.givens): # use the given clues
            if n != 0:
                result.append([x(c,n)])
        return result
    def toSol(self, satSol: List[int]) -> List[int]:
        result = [0]*self.cells
        to_c_n = lambda v : ((v-1)//self.nums, (v-1)%self.nums + 1)
        for v in filter(lambda x : x > 0, satSol):
            c,n = to_c_n(v)
            assert result[c] == 0 # only 1 value assigned to a cell
            result[c] = n
        assert all(n > 0 for n in result) # every cell has a value
        return result

class SatPuzzleSuguruGeneral(SatPuzzleBase):
    '''
    A generalization of of Suguru similar to the generalization of Sudoku. There
    are C > 0 cells divided into areas (specified by a number unique to each
    area). Each area (of size N) must have the numbers 1, 2, ..., N.
    '''
    def __init__(self, cells: int, areas: List[int], givens: List[int]):
        '''
        cells = number of cells, numbered from 0
        areas = list of area numbers assigned to each cell
        givens = list of given values, 0 for no value given
        '''
        assert cells > 0
        assert len(areas) == cells
        assert len(givens) == cells
        area_symbols = set(areas)
        areas_dict: Dict[int,List[int]] = dict() # map symbol to list of cell indexes
        for symb in area_symbols:
            areas_dict[symb] = [i for i,a in enumerate(areas) if a == symb]
        assert all(0 <= n <= len(areas_dict[areas[i]]) for i,n in enumerate(givens))
        self.cells = cells
        self.areas = areas[:]
        self.areasmap = areas_dict
        self.givens = givens[:]
        # variable map for CNF conversion (i,n) -> var (i = cell num)
        self.varmap: Dict[Tuple[int,int],int] = dict()
        self.varmaprev: Dict[int,Tuple[int,int]] = dict() # reverse of above
        last_var = 0 # last variable number used in the next loop
        for i,area in enumerate(areas):
            area_size = len(self.areasmap[area])
            for n in range(1,area_size+1):
                self.varmap[(i,n)] = last_var+n
                self.varmaprev[last_var+n] = (i,n)
            last_var += area_size
    def toCnf(self) -> List[List[int]]:
        '''
        These look very similar to the Sudoku clauses, except the area size and
        amount of possible numbers is constrained by area sizes.
        '''
        result: List[List[int]] = []
        for c in range(self.cells): # each cell
            area_size = len(self.areasmap[self.areas[c]])
            result.append([self.varmap[(c,n)] for n in range(1,area_size+1)]) # has a value
            for n1 in range(1,area_size+1): # and it is unique
                for n2 in range(n1+1,area_size+1):
                    result.append([-self.varmap[(c,n1)],-self.varmap[(c,n2)]])
        for area in self.areasmap: # each area
            cells = self.areasmap[area]
            area_size = len(cells)
            for n in range(1,area_size+1): # has each number
                result.append([self.varmap[(c,n)] for c in cells])
                for i,c1 in enumerate(cells): # that number is in a unique cell (redundant)
                    for c2 in cells[i+1:]:
                        result.append([-self.varmap[(c1,n)],-self.varmap[(c2,n)]])
            for c,n in enumerate(self.givens): # use clues
                if n != 0:
                    result.append([self.varmap[(c,n)]])
        return result
    def toSol(self, satSol: List[int]) -> List[int]:
        result = [0]*self.cells
        for v in filter(lambda x : x > 0, satSol):
            c,n = self.varmaprev[v]
            assert result[c] == 0 # only 1 value assigned to a cell
            result[c] = n
        assert all(n > 0 for n in result)
        return result

class SatPuzzleLatinSquare(SatPuzzleSudokuGeneral):
    '''
    An N x N grid requiring 1,2,...,N in each row/column. Similar to normal
    Sudoku but without the sub blocks.
    Square grid of side length N
    - each row/col must contain 1,2,..,N
    '''
    def __init__(self, givens: List[List[int]]):
        '''
        givens = N x N grid of given numbers, 0 for no value given
        '''
        N = len(givens)
        assert N > 0
        assert len(givens) == N and all(len(row) == N for row in givens) and all(0 <= n <= N for n in sum(givens,[]))
        areas = []
        for r in range(N): # rows
            areas.append(list(range(r*N,r*N+N)))
        for c in range(N): # cols
            areas.append(list(range(c,N*N,N)))
        super().__init__(N*N,N,areas,sum(givens,[]))
    def toSol(self, satSol: List[int]) -> List[List[int]]: # change structure to square
        sol = super().toSol(satSol)
        return [sol[i:i+self.nums] for i in range(0,self.cells,self.nums)]

class SatPuzzleLatinSquareX(SatPuzzleLatinSquare):
    '''
    Latin Square also requiring the diagonals to contain each number once.
    '''
    def __init__(self, givens: List[List[int]]):
        '''
        givens = N x N grid of given numbers, 0 for no value given
        '''
        N = len(givens)
        super().__init__(givens)
        self.areas.append(list(range(0,N*N,N+1)))
        self.areas.append(list(range(N-1,N*N-1,N-1)))

class SatPuzzleSudokuStandard(SatPuzzleLatinSquare):
    '''
    Standard Sudoku is a Latin Square with rectangular sub blocks.
    '''
    def __init__(self, blockR: int, blockC: int, givens: List[List[int]]):
        '''
        blockR = number of rows in each sub block
        blockC = number of columns in each sub block
        givens = N x N grid of given numbers, 0 for no given (N = blockR*blockC)
        '''
        assert blockR > 0 and blockC > 0
        N = blockR*blockC
        assert len(givens) == N
        super().__init__(givens)
        for r in range(blockC): # blocks
            for c in range(blockR):
                self.areas.append([(r*blockR+rr)*N+(c*blockC+cc) for rr in range(blockR) for cc in range(blockC)])
        self.blockR = blockR
        self.blockC = blockC

class SatPuzzleSudokuX(SatPuzzleSudokuStandard):
    '''
    Standard Sudoku also requiring the diagonals to contain each number once.
    '''
    def __init__(self, blockR: int, blockC: int, givens: List[List[int]]):
        '''
        blockR = number of rows in each sub block
        blockC = number of columns in each sub block
        givens = N x N grid of given numbers, 0 for no given (N = blockR*blockC)
        '''
        super().__init__(blockR,blockC,givens)
        N = blockR*blockC
        self.areas.append(list(range(0,N*N,N+1)))
        self.areas.append(list(range(N-1,N*N-1,N-1)))

class SatPuzzleSudokuJigsaw(SatPuzzleLatinSquare):
    '''
    Sudoku with irregularly shaped areas.
    '''
    def __init__(self, givens: List[List[int]], areas: List[List[int]]):
        '''
        givens = N x N grid of given numbers, 0 for no given
        areas = N x N grid of area assigned to each number (each area is a unique integer)
        '''
        N = len(givens)
        assert N > 0
        assert len(areas) == N
        assert all(len(row) == N for row in areas)
        super().__init__(givens)
        for symb in set(sum(areas,[])): # add areas
            self.areas.append([i for i,a in enumerate(sum(areas,[])) if a == symb])

class SatPuzzleSudokuJigsawX(SatPuzzleSudokuJigsaw):
    '''
    Jigsaw sudoku with main diagonals
    '''
    def __init__(self, givens: List[List[int]], areas: List[List[int]]):
        '''
        givens = N x N grid of given numbers, 0 for no given
        areas = N x N grid of area assigned to each number (each area is a unique integer)
        '''
        super().__init__(givens,areas)
        N = len(givens)
        self.areas.append(list(range(0,N*N,N+1)))
        self.areas.append(list(range(N-1,N*N-1,N-1)))

class SatPuzzleSudokuOverlap(SatPuzzleSudokuGeneral):
    '''
    A generalization of Sudoku variants that overlap several standard sudoku
    grids overlapping on rectangular sub blocks.
    '''
    def __init__(self, blockR: int, blockC: int, givens: List[List[int]], corners: List[Tuple[int,int]]):
        '''
        blockR = number of rows in each sub block
        blockC = number of columns in each sub block
        givens = grid of given values, 0 for no value, should have 0 in unused areas
        corners = list of (r,c) positions (with (0,0) being the top left)
            representing the top left corners of the standard sudoku puzzles
        '''
        N = blockR*blockC
        assert all(pr%blockR == 0 and pc%blockC == 0 for pr,pc in corners)
        brows = max(p[0] for p in corners)//blockR + blockC
        bcols = max(p[1] for p in corners)//blockC + blockR
        R = brows*blockR
        C = bcols*blockC
        self.rows = R
        self.cols = C
        self.blockR = blockR
        self.blockC = blockC
        self.corners = corners[:]
        assert len(givens) == R and all(len(row) == C for row in givens)
        areas = []
        self.blockmap = [[False]*bcols for _ in range(brows)]
        for pr,pc in corners: # for each (overlapping) standard sudoku
            for r in range(pr,pr+N): # rows
                areas.append(list(range(r*C+pc,r*C+pc+N)))
            for c in range(pc,pc+N): # cols
                areas.append(list(range(pr*C+c,(pr+N)*C+c,C)))
            for br in range(blockC): # areas for this standard puzzle
                for bc in range(blockR):
                    brow,bcol = pr//blockR+br, pc//blockC+bc
                    if not self.blockmap[brow][bcol]: # no area for this block yet
                        self.blockmap[brow][bcol] = True
                        areas.append([(brow*blockR+rr)*C+(bcol*blockC+cc) for rr in range(blockR) for cc in range(blockC)])
        # set dummy number in unused areas
        givens2 = [row[:] for row in givens]
        for brow,blockmaprow in enumerate(self.blockmap):
            for bcol,used in enumerate(blockmaprow):
                if not used:
                    for r in range(blockR):
                        for c in range(blockC):
                            assert givens[brow*blockR+r][bcol*blockC+c] == 0 # should provide 0 in unused space
                            givens2[brow*blockR+r][bcol*blockC+c] = 1
        super().__init__(R*C,N,areas,sum(givens2,[]))
    def toSol(self, satSol: List[int]) -> List[List[int]]:
        sol = super().toSol(satSol)
        gridSol = [sol[i:i+self.cols] for i in range(0,self.rows*self.cols,self.cols)]
        # set unused areas to zeroes
        for brow,blockmaprow in enumerate(self.blockmap):
            for bcol,used in enumerate(blockmaprow):
                if not used:
                    for r in range(self.blockR):
                        for c in range(self.blockC):
                            gridSol[brow*self.blockR+r][bcol*self.blockC+c] = 0
        return gridSol

class SatPuzzleSudokuButterfly(SatPuzzleSudokuOverlap):
    '''
    Overlapped standard (size 9x9) Sudokus in the following arrangement:
        ABXX
        CDXX
        XXXX
        XXXX
    Each letter is a sub block and ABCD ore the top left corners of the standard Sudokus.
    This class only supports 3x3 sub blocks.
    '''
    def __init__(self, givens: List[List[int]]):
        '''
        givens = 12x12 grid
        '''
        super().__init__(3,3,givens,[(0,0),(0,3),(3,0),(3,3)])

class SatPuzzleSudokuCluelessBase(SatPuzzleSudokuOverlap):
    '''
    Shared part of the Clueless Sudoku types (9 non overlapping standard grids)
    AXXBXXCXX
    XXXXXXXXX
    XXXXXXXXX
    DXXEXXFXX
    XXXXXXXXX
    XXXXXXXXX
    GXXHXXIXX
    XXXXXXXXX
    XXXXXXXXX
    Each letter is a sub block and ABCDEFGHI are the top left corners of the standard Sudokus.
    This class only supports 3x3 sub blocks.
    '''
    def __init__(self, givens: List[List[int]]):
        '''
        givens = 27x27 grid
        '''
        super().__init__(3,3,givens,[(0,0),(0,9),(0,18),(9,0),(9,9),(9,18),(18,0),(18,9),(18,18)])

class SatPuzzleSudokuClueless1(SatPuzzleSudokuCluelessBase):
    '''
    The 10th grid is made of the centers of the 81 sub blocks.
    '''
    def __init__(self, givens: List[List[int]]):
        super().__init__(givens)
        for r in range(1,27,3):
            self.areas.append(list(range(27*r+1,27*r+27,3)))
        for c in range(1,27,3):
            self.areas.append(list(range(27*1+c,27*27+c,27*3)))
        for r in range(1,27,9):
            for c in range(1,27,9):
                self.areas.append([27*(r+rr)+(c+cc) for rr in range(0,9,3) for cc in range(0,9,3)])

class SatPuzzleSudokuClueless2(SatPuzzleSudokuCluelessBase):
    '''
    The 10th grid is made of the center blocks of each sub puzzle.
    '''
    def __init__(self, givens: List[List[int]]):
        super().__init__(givens)
        indexes = [3,4,5,12,13,14,21,22,23]
        for i in indexes:
            self.areas.append([27*i+c for c in indexes]) # row
            self.areas.append([27*r+i for r in indexes]) # col
        # blocks are redundant since they are part of other sub puzzles

class SatPuzzleSudokuFlower(SatPuzzleSudokuOverlap):
    '''
    Overlapped standard (size 9x9) Sudokus in the following arrangement:
         AXX
        BXCXX
        XDXXX
        XXXXX
         XXX
    Each letter is a sub block and ABCD are the top left corners of the standard Sudokus.
    This class only supports 3x3 sub blocks.
    '''
    def __init__(self, givens: List[List[int]]):
        '''
        givens = 15x15 grid
        '''
        super().__init__(3,3,givens,[(0,3),(3,0),(3,3),(3,6),(6,3)])

class SatPuzzleSudokuGattai8(SatPuzzleSudokuOverlap):
    '''
    Overlapped standard (size 9x9) Sudokus in the following arrangement:
        AXX BXX CXX
        XXX XXX XXX
        XXDXXXEXXXX
          XXX XXX
        FXXXGXXXHXX
        XXX XXX XXX
        XXX XXX XXX
    Each letter is a sub block and ABCDEFGH are the top left corners of the standard Sudokus.
    This class only supports 3x3 sub blocks.
    '''
    def __init__(self, givens: List[List[int]]):
        '''
        givens = 21x33 grid
        '''
        super().__init__(3,3,givens,[(0,0),(0,12),(0,24),(6,6),(6,18),(12,0),(12,12),(12,24)])

class SatPuzzleSudokuConsecutive(SatPuzzleLatinSquare):
    '''
    Latin Square with edge markings for orthogonally adjacent cells that must
    have consecutive values. This does not have sub blocks like normal Sudoku.
    '''
    def __init__(self, givens: List[List[int]], consec_pairs: Set[Tuple[Tuple[int,int],Tuple[int,int]]]):
        '''
        givens = N x N grid of given numbers
        consec_pairs = list of (r1,c1),(r2,c2) specifying 2 cells that must have consecutive values
            these should only be orthogonally adjacent
        '''
        N = len(givens)
        super().__init__(givens)
        assert all(0 <= r1 < N and 0 <= c1 < N and 0 <= r2 < N and 0 <= c2 < N for (r1,c1),(r2,c2) in consec_pairs)
        self.consec_pairs = set(p for p in consec_pairs)
    def toCnf(self) -> List[List[int]]: # extend to add extra constraints
        '''
        Add constraints for orthogonally adjacent cells
        Let c1,c2 be the cells and x(c,n) mean the variable for n assigned to
        cell c. The possible assignments are 2 distinct values to these cells.
        If we constrain the values to be consecutive, then we add the clauses:
        - not x(c1,a) or not x(c2,b) (for each non consecutive pair a,b)
        If c1,c2 are assigned consecutive values, then all these clauses will be
        true. Otherwise, one of them will be false. For requiring 2 cells to be
        non consecutive, use clauses of the same form for consecutive pairs a,b.
        The reasoning is similar.
        '''
        result = super().toCnf()
        N = self.nums
        x = lambda r,c,n : 1 + (r*N+c)*N + (n-1) # get variable number
        consec = [(a,b) for a in range(1,N+1) for b in range(1,N+1) if abs(a-b) == 1]
        nonconsec = [(a,b) for a in range(1,N+1) for b in range(1,N+1) if abs(a-b) > 1]
        for r in range(N-1): # horizontal markers - (r,c) and (r+1,c)
            for c in range(N):
                # if must be consecutive, then for any pair of nonconsecutive numbers, either cell is not equal to it
                # if must not be consecutive, then for any pair of consecutive numbers, either cell is not equal to it
                for a,b in nonconsec if ((r,c),(r+1,c)) in self.consec_pairs else consec:
                    result.append([-x(r,c,a),-x(r+1,c,b)])
        for r in range(N): # vertical markers - (r,c) and (r,c+1)
            for c in range(N-1):
                for a,b in nonconsec if ((r,c),(r,c+1)) in self.consec_pairs else consec:
                    result.append([-x(r,c,a),-x(r,c+1,b)])
        return result

class SatPuzzleSudokuKropki(SatPuzzleSudokuStandard):
    '''
    Standard Sudoku with some circles on the edges. White circles mean the 2
    values must be consecutive. Black circles mean one value is double of the
    other. No circle means the numbers are neither.
    '''
    def __init__(self, blockR: int, blockC, white: Set[Tuple[Tuple[int,int],Tuple[int,int]]], black: Set[Tuple[Tuple[int,int],Tuple[int,int]]]):
        '''
        blockR = rows per sub block
        blockC = cols per sub block
        white = set of (r1,c1),(r2,c2) cell pairs with a white circle
        black = set of (r1,c1),(r2,c2) cell pairs with a black circle
        '''
        N = blockR*blockC
        super().__init__(blockR,blockC,[[0]*N for _ in range(N)])
        assert all(0 <= r1 < N and 0 <= c1 < N and 0 <= r2 < N and 0 <= c2 < N for (r1,c1),(r2,c2) in white)
        assert all(0 <= r1 < N and 0 <= c1 < N and 0 <= r2 < N and 0 <= c2 < N for (r1,c1),(r2,c2) in black)
        assert white & black == set()
        self.white = set(p for p in white)
        self.black = set(p for p in black)
    def toCnf(self) -> List[List[int]]: # extend to add extra constraints
        '''
        These constraints are handled similarly to those in Consecutive Sudoku.
        For each pair of cells c1,c2 and each pair a,b of (distinct) cell values
        that are not allowed in these 2 cells, add the clause:
        - not x(c1,a) or not x(c2,b)
        '''
        result = super().toCnf()
        N = self.nums
        x = lambda r,c,n : 1 + (r*N+c)*N + (n-1) # get variable number
        # sets of possible pairs
        consec = set((a,b) for a in range(1,N+1) for b in range(1,N+1) if abs(a-b) == 1)
        double = set((a,b) for a in range(1,N+1) for b in range(1,N+1) if a == 2*b or a*2 == b)
        neither = set((a,b) for a in range(1,N+1) for b in range(1,N+1) if a != b and ((a,b) not in consec) and ((a,b) not in double))
        all_pairs = set((a,b) for a in range(1,N+1) for b in range(1,N+1) if a != b)
        # complements for adding clauses
        c_consec = all_pairs - consec
        c_double = all_pairs - double
        c_neither = all_pairs - neither
        for r in range(N-1): # circles on horizontal (vertical pair)
            for c in range(N):
                pairs = c_consec if ((r,c),(r+1,c)) in self.white else (c_double if ((r,c),(r+1,c)) in self.black else c_neither)
                for a,b in pairs:
                    result.append([-x(r,c,a),-x(r+1,c,b)])
        for r in range(N): # circles on vertical (horizontal pair)
            for c in range(N-1):
                pairs = c_consec if ((r,c),(r,c+1)) in self.white else (c_double if ((r,c),(r,c+1)) in self.black else c_neither)
                for a,b in pairs:
                    result.append([-x(r,c,a),-x(r,c+1,b)])
        return result

class SatPuzzleSudokuMagicNumberX(SatPuzzleLatinSquareX):
    '''
    Latin Square with thick lines between cells indicating they must sum to the
    given magic number.
    '''
    def __init__(self, givens: List[List[int]], magic: int, pairs: Set[Tuple[Tuple[int,int],Tuple[int,int]]]):
        '''
        givens = N x N grid of given values, 0 for no given value
        magic = magic number
        pairs = set of (r1,c1),(r2,c2) cell pairs that must sum to magic number
        '''
        N = len(givens)
        super().__init__(givens)
        self.magic = magic
        assert all(0 <= r1 < N and 0 <= c1 < N and 0 <= r2 < N and 0 <= c2 < N for (r1,c1),(r2,c2) in pairs)
        self.pairs = set(p for p in pairs)
    def toCnf(self) -> List[List[int]]:
        '''
        Add constraints for orthogonally adjacent cells similar to in
        Consecutive Sudoku where clauses are added for each pair that is not
        allowed in the 2 cells.
        '''
        result = super().toCnf()
        N = self.nums
        x = lambda r,c,n : 1 + (r*N+c)*N + (n-1) # get variable number
        not_magic_sum = [(a,b) for a in range(1,N+1) for b in range(1,N+1) if a != b and a+b != self.magic]
        for (r1,c1),(r2,c2) in self.pairs:
            for a,b in not_magic_sum: # at least 1 of these must not be set
                result.append([-x(r1,c1,a),-x(r2,c2,b)])
        return result

class SatPuzzleSudokuSamurai(SatPuzzleSudokuOverlap):
    '''
    Overlapped standard (size 9x9) Sudokus in the following arrangement:
    AXX BXX
    XXX XXX
    XXCXXXX
      XXX
    DXXXEXX
    XXX XXX
    XXX XXX
    Each letter is a sub block and ABCDE are the top left corners of the standard Sudokus.
    This class only supports 3x3 sub blocks.
    '''
    def __init__(self, givens: List[List[int]]):
        '''
        givens = 21x21 grid
        '''
        super().__init__(3,3,givens,[(0,0),(0,12),(6,6),(12,0),(12,12)])

class SatPuzzleSudokuOddEven(SatPuzzleSudokuStandard):
    '''
    Sudoku variant that marks all odd numbers with circles and leaves all even
    numbers unmarked.
    '''
    def __init__(self, blockR: int, blockC: int, givens: List[List[int]], odds: List[List[bool]]):
        '''
        blockR = rows per sub block
        blockC = cols per sub block
        givens = N x N grid (N = blockR*blockC)
        odds = boolean array, True for numbers marked with circles
        '''
        super().__init__(blockR,blockC,givens)
        N = len(givens)
        assert len(odds) == N and all(len(row) == N for row in odds)
        self.odds = [row[:] for row in odds]
    def toCnf(self) -> List[List[int]]:
        '''
        Add constraints for odd and even cells. These are expressed as negations
        of assigning the opposite parity to cells.
        '''
        result = super().toCnf()
        N = self.nums
        x = lambda r,c,n : 1 + (r*N+c)*N + (n-1) # get variable number
        for r,row in enumerate(self.odds):
            for c,odd in enumerate(row):
                if odd:
                    for n in range(2,N+1,2): # cannot be even
                        result.append([-x(r,c,n)])
                else:
                    for n in range(1,N+1,2): # cannot be odd
                        result.append([-x(r,c,n)])
        return result

class SatPuzzleSudokuOddEvenSamurai(SatPuzzleSudokuSamurai):
    '''
    Odd Even Sudoku in the Samurai overlapping layout.
    '''
    def __init__(self, givens: List[List[int]], odds: List[List[bool]]):
        '''
        givens = 15x15 grid
        odds = 15x15 grid
        '''
        super().__init__(givens)
        assert len(odds) == 21 and all(len(row) == 21 for row in odds)
        self.odds = [row[:] for row in odds]
    def toCnf(self) -> List[List[int]]:
        '''
        Handle these constraints with a more restrictive clause of possible
        values which makes the all value clauses redundant.
        '''
        result = super().toCnf()
        x = lambda r,c,n : 1 + (r*21+c)*9 + (n-1)
        unused_blocks = set([(0,3),(1,3),(3,0),(3,1),(3,5),(3,6),(5,3),(6,3)])
        for r,row in enumerate(self.odds): # constrain number parity in used blocks
            for c,odd in enumerate(row):
                if (r//self.blockR,c//self.blockC) in unused_blocks:
                    continue
                elif odd: # cell must be odd
                    result.append([x(r,c,n) for n in range(1,10,2)])
                else: # cell must be even
                    result.append([x(r,c,n) for n in range(2,10,2)])
        return result

class SatPuzzleSudokuMarginalSum(SatPuzzleSudokuStandard):
    '''
    Sudoku grid with no given cell clues. The clues given are the sum of the
    first 3 numbers from the edges. This class extends to other sub block sizes.
    '''
    def __init__(self, blockR: int, blockC: int, top: List[int], bottom: List[int], left: List[int], right: List[int]):
        '''
        blockR = rows per sub block
        blockC = cols per sub block
        top = sum of top blockR numbers in each column
        bottom = sum of bottom blockR numbers in each column
        left = sum of left blockC numbers in each row
        righth = sum of right blockC numbers in each row
        '''
        N = blockR*blockC
        super().__init__(blockR,blockC,[[0]*N for _ in range(N)])
        assert len(top) == len(bottom) == len(left) == len(right) == N
        self.top = top[:]
        self.bottom = bottom[:]
        self.left = left[:]
        self.right = right[:]
    def toCnf(self) -> List[List[int]]:
        '''
        The marginal sum constraints can be represented as clauses for the
        permutations of numbers not allowed (not n1 or not n2 or ...). If the
        numbers are assigned one of the disallowed permutations, there will be
        a false clause. Otherwise, all these clauses will be true. This grows
        exponentially in the blockR and blockC parameters so it is not a proper
        reduction.
        TODO research ways to make this reduction polynomial time and space
        '''
        result = super().toCnf()
        N = self.nums
        br = self.blockR
        bc = self.blockC
        x = lambda r,c,n : 1 + (r*N+c)*N + (n-1)
        # generator for all permutations of some of the numbers
        def recur(size: int, partial: List[int] = []) -> Generator[List[int],None,None]:
            if len(partial) == size:
                yield partial[:]
            else:
                for n in range(1,N+1):
                    if n not in partial:
                        yield from recur(size,partial+[n])
        # these clauses for the marginal sum grow exponentially with puzzle size
        for i in range(N):
            for perm in recur(br): # top
                if sum(perm) != self.top[i]:
                    result.append([-x(j,i,perm[j]) for j in range(br)])
            for perm in recur(br): # bottom
                if sum(perm) != self.bottom[i]:
                    result.append([-x(N-br+j,i,perm[j]) for j in range(br)])
            for perm in recur(bc): # left
                if sum(perm) != self.left[i]:
                    result.append([-x(i,j,perm[j]) for j in range(bc)])
            for perm in recur(bc): # right
                if sum(perm) != self.right[i]:
                    result.append([-x(i,N-bc+j,perm[j]) for j in range(bc)])
        return result

class SatPuzzleSudokuShogun(SatPuzzleSudokuOverlap):
    '''
    Overlapped standard (size 9x9) Sudokus in the following arrangement:
    AXX BXX CXX DXX
    XXX XXX XXX XXX
    XXEXXXFXXXGXXXX
      XXX XXX XXX
    HXXXIXXXJXXXKXX
    XXX XXX XXX XXX
    XXX XXX XXX XXX
    Each letter is a sub block and ABCDEFGHIJK are the top left corners of the standard Sudokus.
    This class only supports 3x3 sub blocks.
    '''
    def __init__(self, givens: List[List[int]]):
        '''
        givens = 21x45 grid
        '''
        super().__init__(3,3,givens,[(0,0),(0,12),(0,24),(0,36),(6,6),(6,18),(6,30),(12,0),(12,12),(12,24),(12,36)])

class SatPuzzleSudokuSohei(SatPuzzleSudokuOverlap):
    '''
    Overlapped standard (size 9x9) Sudokus in the following arrangement:
      AXX
      XXX
    BXXXCXX
    XXX XXX
    XXDXXXX
      XXX
      XXX
    Each letter is a sub block and ABCD are the top left corners of the standard Sudokus.
    This class only supports 3x3 sub blocks.
    '''
    def __init__(self, givens: List[List[int]]):
        '''
        givens = 21x21 grid
        '''
        super().__init__(3,3,givens,[(0,6),(6,0),(6,12),(12,6)])

class SatPuzzleSudokuSumo(SatPuzzleSudokuOverlap):
    '''
    Overlapped standard (size 9x9) Sudokus in the following arrangement:
    AXX BXX CXX
    XXX XXX XXX
    XXDXXXEXXXX
      XXX XXX
    FXXXGXXXHXX
    XXX XXX XXX
    XXIXXXJXXXX
      XXX XXX
    KXXXLXXXMXX
    XXX XXX XXX
    XXX XXX XXX
    Each letter is a sub block and ABCDEFGHIJKLM are the top left corners of the standard Sudokus.
    This class only supports 3x3 sub blocks.
    '''
    def __init__(self, givens: List[List[int]]):
        '''
        givens = 33x33 grid
        '''
        super().__init__(3,3,givens,[(0,0),(0,12),(0,24),(6,6),(6,18),(12,0),(12,12),(12,24),(18,6),(18,18),(24,0),(24,12),(24,24)])

class SatPuzzleSudokuWindmill(SatPuzzleSudokuOverlap):
    '''
    Overlapped standard (size 9x9) Sudokus in the following arrangement:
     AXX
     XXXBXX
     XCXXXX
    DXXXXXX
    XXXEXX
    XXXXXX
       XXX
    Each letter is a sub block and ABCDE are the top left corners of the standard Sudokus.
    This class only supports 3x3 sub blocks.
    '''
    def __init__(self, givens: List[List[int]]):
        '''
        givens = 21x21 grid
        '''
        super().__init__(3,3,givens,[(0,3),(3,12),(6,6),(9,0),(12,9)])

class SatPuzzleSudokuComparison(SatPuzzleSudokuStandard):
    '''
    Empty Sudoku grid with all cell borders inside the sub blocks showing a less
    than relation between adjacent cell pairs (within the same sub block).
    '''
    def __init__(self, blockR: int, blockC: int, relations: Set[Tuple[Tuple[int,int],Tuple[int,int]]]):
        '''
        blockR = rows per sub block
        blockC = cols per sub block
        relations = cell pairs (r1,c1),(r2,c2) where r1,c1 value < r2,c2 value
        '''
        N = blockR*blockC
        super().__init__(blockR,blockC,[[0]*(N) for _ in range(N)])
        assert all(0 <= r1 < N and 0 <= c1 < N and 0 <= r2 < N and 0 <= c2 < N for (r1,c1),(r2,c2) in relations)
        self.relations = set(p for p in relations)
    def toCnf(self) -> List[List[int]]:
        '''
        For each cell pair c1,c2 with c1 value < c2 value, add clauses:
        not x(c1,b) or not x(c2,a) for 1 <= a < b <= N
        This disallows all assignments that violate the less than constraint.
        '''
        result = super().toCnf()
        N = self.nums
        x = lambda r,c,n : 1 + (r*N+c)*N + (n-1) # get variable number
        for (r1,c1),(r2,c2) in self.relations:
            # must have a not equal in invalid pairs (r1,c1) > (r2,c2)
            for a in range(1,N+1):
                for b in range(a+1,N+1):
                    result.append([-x(r1,c1,b),-x(r2,c2,a)])
        return result

class SatPuzzleSuguruStandard(SatPuzzleSuguruGeneral):
    '''
    An M x N grid divided into areas. Each area of size A must contain the
    numbers 1,2,..,N. The same number cannot be adjacent or diagonally adjacent.
    '''
    def __init__(self, givens: List[List[int]], areas: List[List[int]]):
        '''
        givens = grid of given numbers, 0 for none, otherwise they must not be
            larger than their area size
        areas = area for each cell, each area represented by a unique symbol
        '''
        R = len(givens)
        C = len(givens[0])
        self.rows = R
        self.cols = C
        assert R > 0 and C > 0
        assert all(len(row) == C for row in givens)
        assert len(areas) == R and all(len(row) == C for row in areas)
        super().__init__(R*C,sum(areas,[]),sum(givens,[]))
    def toCnf(self) -> List[List[int]]:
        '''
        For each cell c1, add up to 8 constraints for neighboring cells c2:
        - not x(c1,n) or not x(c2,n)
        This needs to be done for all n limited by the size of the area(s) c1
        and c2 are in.
        '''
        result = super().toCnf()
        for r in range(self.rows):
            for c in range(self.cols):
                # for the 8 cells around it
                for dr in range(-1,2):
                    for dc in range(-1,2):
                        if dr == 0 and dc == 0:
                            continue
                        rr,cc = r+dr,c+dc
                        if rr < 0 or rr >= self.rows or cc < 0 or cc >= self.cols:
                            continue # off grid
                        i1 = r*self.cols+c
                        i2 = rr*self.cols+cc
                        area1 = self.areas[i1]
                        area2 = self.areas[i2]
                        area1size = len(self.areasmap[area1])
                        area2size = len(self.areasmap[area2])
                        # n at (r,c) implies n not at at (rr,cc)
                        for n in range(1,min(area1size+1,area2size+1)):
                            result.append([-self.varmap[(i1,n)],-self.varmap[(i2,n)]])
        return result
    def toSol(self, satSol: List[int]) -> List[List[int]]:
        sol = super().toSol(satSol)
        return [sol[i:i+self.cols] for i in range(0,self.rows*self.cols,self.cols)]

class SatPuzzleHakyuu(SatPuzzleSuguruGeneral):
    '''
    A Suguru variant that uses a different rule for restricting number
    placement. Within each row/col, all occurrences of a number n must have at
    least n cells between them.
    '''
    def __init__(self, givens: List[List[int]], areas: List[List[int]]):
        '''
        givens = grid of given numbers, 0 for none, otherwise they must not be
            larger than their area size
        areas = area for each cell, each area represented by a unique symbol
        '''
        R = len(givens)
        C = len(givens[0])
        self.rows = R
        self.cols = C
        assert R > 0 and C > 0
        assert all(len(row) == C for row in givens)
        assert len(areas) == R and all(len(row) == C for row in areas)
        super().__init__(R*C,sum(areas,[]),sum(givens,[]))
    def toCnf(self) -> List[List[int]]:
        '''
        For each cell c1 and a number n, x(c1,n) implies n is not assigned to
        some cells in the same row/col depending on n. For each cell c2 from 1
        to n away from c1 in the same row/col, add this constraint if n is <=
        the size of the area(s) of c1 and c2:
        - not x(c1,n) or not x(c2,n)
        '''
        result = super().toCnf()
        for r in range(self.rows):
            for c in range(self.cols):
                # n at (r,c) implies n not within n cells in orthogonal directions
                i1 = r*self.cols+c
                area1 = self.areas[i1]
                area1size = len(self.areasmap[area1])
                for n in range(1,area1size+1):
                    for d in range(1,n+1): # distance from r,c
                        for rr,cc in [(r+d,c),(r-d,c),(r,c+d),(r,c-d)]:
                            if rr < 0 or rr >= self.rows or cc < 0 or cc >= self.cols:
                                continue # off grid
                            i2 = rr*self.cols+cc
                            area2 = self.areas[i2]
                            area2size = len(self.areasmap[area2])
                            if n <= area2size: # n at (r,c) implies n not at (rr,cc)
                                result.append([-self.varmap[i1,n],-self.varmap[i2,n]])
        return result
    def toSol(self, satSol: List[int]) -> List[List[int]]:
        sol = super().toSol(satSol)
        return [sol[i:i+self.cols] for i in range(0,self.rows*self.cols,self.cols)]

class SatPuzzleSukaku(SatPuzzleSudokuStandard):
    '''
    Standard Sudoku with a candidate set given for each cell.
    '''
    def __init__(self, blockR: int, blockC: int, candidates: List[List[List[int]]]):
        '''
        blockR = rows per sub block
        blockC = cols per sub block
        candidates = N x N grid of candidate list for each cell
        '''
        N = blockR*blockC
        super().__init__(blockR,blockC,[[0]*N for _ in range(N)])
        assert len(candidates) == N and all(len(row) == N for row in candidates)
        assert all(all(len(set(values)) == len(values) and all(1 <= value <= N for value in values) for values in row) for row in candidates)
        self.candidates = [[values[:] for values in row] for row in candidates]
    def toCnf(self) -> List[List[int]]:
        '''
        The additional constraints added constrain cell values to those in the
        candidate sets. This makes the original constraints redundant for
        assigning a value to a cell.
        '''
        result = super().toCnf()
        N = self.nums
        x = lambda r,c,n : 1 + (r*N+c)*N + (n-1)
        for r in range(N):
            for c in range(N):
                result.append([x(r,c,n) for n in self.candidates[r][c]])
        return result

class SatPuzzleSukakuJigsaw(SatPuzzleSudokuJigsaw):
    '''
    Sukaku, but with irregularly shaped areas instead of rectangular.
    '''
    def __init__(self, areas: List[List[int]], candidates: List[List[List[int]]]):
        '''
        areas = N x N grid of cell areas, each area is a unique symbol
        candidates = N x N grid of candidate list for each cell
        '''
        N = len(areas)
        super().__init__([[0]*N for _ in range(N)],areas)
        assert len(candidates) == N and all(len(row) == N for row in candidates)
        assert all(all(len(set(values)) == len(values) and all(1 <= value <= N for value in values) for values in row) for row in candidates)
        self.candidates = [[values[:] for values in row] for row in candidates]
    def toCnf(self) -> List[List[int]]:
        '''
        The constraints here are handled exactly the same way as they are in
        regular Sukaku.
        '''
        result = super().toCnf()
        N = self.nums
        x = lambda r,c,n : 1 + (r*N+c)*N + (n-1)
        for r in range(N):
            for c in range(N):
                result.append([x(r,c,n) for n in self.candidates[r][c]])
        return result
