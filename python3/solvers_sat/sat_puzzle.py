from itertools import chain
import pycosat
from typing import Any, Iterator, List

class SatPuzzleBase:
    def toCnf(self) -> List[List[int]]:
        assert 0, 'not implemented'
        return []
    def toSol(self, satSol: List[int]) -> Any:
        assert 0, 'not implemented'
    def cnfSolve(self) -> List[int]:
        result = pycosat.solve(self.toCnf())
        assert result != 'UNKNOWN'
        return [] if result == 'UNSAT' else result
    def cnfSolveAll(self) -> Iterator[List[int]]:
        return pycosat.itersolve(self.toCnf())
    def solve(self) -> Any:
        return self.toSol(self.cnfSolve())
    def solveAll(self) -> Iterator[Any]:
        return map(self.toSol,self.cnfSolveAll())

class SatPuzzleSudokuGeneral(SatPuzzleBase):
    '''
    Generalization of sudoku
    Puzzle is defined by some number of cells, a positive integer N, and areas
    (set of N cells) which must contain 1,2,..,N each exactly once.
    '''
    def __init__(self, cells: int, nums: int, areas: List[List[int]], givens: List[int]):
        '''
        cells = number of cells, numbered starting from 0
        nums = symbols in puzzle, 1, 2, .., N
        areas = list of cells constrained to contain 1..N, each of size N
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
        variables: x_{c,n} (0 <= c < cells, 1 <= n <= nums)
        constraints:
        - each cell has >= 1 value
        - each cell has <= 1 value
        - each area contains each number
        - use the given clues
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
                for i,c1 in enumerate(area): # for any 2 cells, one does not have n ()
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

class SatPuzzleLatinSquare(SatPuzzleSudokuGeneral):
    '''
    Square grid of side length N
    - each row/col must contain 1,2,..,N
    '''
    def __init__(self, size: int, givens: List[List[int]]):
        assert size > 0
        N = size
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
    Latin square with diagonals
    '''
    def __init__(self, size: int, givens: List[List[int]]):
        N = size
        super().__init__(N,givens)
        self.areas.append(list(range(0,N*N,N+1)))
        self.areas.append(list(range(N-1,N*N-1,N-1)))

class SatPuzzleSudokuStandard(SatPuzzleLatinSquare):
    '''
    Extends latin square by adding rectangular blocks
    '''
    def __init__(self, blockR: int, blockC: int, givens: List[List[int]]):
        assert blockR > 0 and blockC > 0
        N = blockR*blockC
        super().__init__(N,givens)
        for r in range(blockC): # blocks
            for c in range(blockR):
                self.areas.append([(r*blockR+rr)*N+(c*blockC+cc) for rr in range(blockR) for cc in range(blockC)])

class SatPuzzleSudokuX(SatPuzzleSudokuStandard):
    '''
    Also requires the main diagonals to contain 1,2,..,N
    '''
    def __init__(self, blockR: int, blockC: int, givens: List[List[int]]):
        super().__init__(blockR,blockC,givens)
        N = blockR*blockC
        self.areas.append(list(range(0,N*N,N+1)))
        self.areas.append(list(range(N-1,N*N-1,N-1)))

class SatPuzzleSudokuJigsaw(SatPuzzleLatinSquare):
    '''
    Divides grid into (irregular) areas
    '''
    def __init__(self, nums: int, givens: List[List[int]], areas: List[List[int]]):
        assert nums > 0
        super().__init__(nums,givens)
        for symb in set(sum(areas,[])): # add areas
            self.areas.append([i for i,a in enumerate(sum(areas,[])) if a == symb])

class SatPuzzleSudokuJigsawX(SatPuzzleSudokuJigsaw):
    '''
    Jigsaw sudoku with main diagonals
    '''
    def __init__(self, nums: int, givens: List[List[int]], areas: List[List[int]]):
        super().__init__(nums,givens,areas)
        N = nums
        self.areas.append(list(range(0,N*N,N+1)))
        self.areas.append(list(range(N-1,N*N-1,N-1)))

class SatPuzzleSudokuButterfly(SatPuzzleSudokuGeneral):
    '''
    12x12 grid with 4 9x9 standard sudoku
    only supporting standard type for now
    '''
    def __init__(self, givens: List[List[int]]):
        assert len(givens) == 12 and all(len(row) == 12 for row in givens)
        areas = []
        for r in range(12):
            areas.append(list(range(12*r+0,12*r+9))) # left
            areas.append(list(range(12*r+3,12*r+12))) # right
        for c in range(12):
            areas.append(list(range(0*12+c,9*12,12))) # top
            areas.append(list(range(3*12+c,12*12,12))) # bottom
        for r in range(0,12,3): # blocks
            for c in range(0,12,3):
                areas.append([12*(r+rr)+(c+cc) for rr in range(3) for cc in range(3)])
        super().__init__(12*12,9,areas,sum(givens,[]))
    def toSol(self, satSol: List[int]) -> List[List[int]]:
        sol = super().toSol(satSol)
        return [sol[i:i+12] for i in range(0,12*12,12)]

class SatPuzzleSudokuCluelessBase(SatPuzzleSudokuGeneral):
    '''
    Shared part of the clueless sudoku types (9 non overlapping standard grids)
    '''
    def __init__(self, givens: List[List[int]]):
        assert len(givens) == 27 and all(len(row) == 27 for row in givens)
        areas = []
        for r in range(27):
            areas.append(list(range(27*r+0,27*r+9))) # left
            areas.append(list(range(27*r+9,27*r+18))) # middle
            areas.append(list(range(27*r+18,27*r+27))) # right
        for c in range(27):
            areas.append(list(range(0*27+c,9*27,27))) # top
            areas.append(list(range(9*27+c,18*27,27))) # middle
            areas.append(list(range(18*27+c,27*27,27))) # botto
        for r in range(0,27,3): # blocks
            for c in range(0,27,3):
                areas.append([27*(r+rr)+(c+cc) for rr in range(3) for cc in range(3)])
        super().__init__(27*27,9,areas,sum(givens,[]))
    def toSol(self, satSol: List[int]) -> List[List[int]]:
        sol = super().toSol(satSol)
        return [sol[i:i+27] for i in range(0,27*27,27)]

class SatPuzzleSudokuClueless1(SatPuzzleSudokuCluelessBase):
    '''
    Special grid is the center of the 81 sub blocks
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
    Special grid is the center blocks of each grid
    '''
    def __init__(self, givens: List[List[int]]):
        super().__init__(givens)
        indexes = [3,4,5,12,13,14,21,22,23]
        for i in indexes:
            self.areas.append([27*i+c for c in indexes]) # row
            self.areas.append([27*r+i for r in indexes]) # col
        # blocks are redundant since they are part of other sub puzzles

class SatPuzzleSudokuFlower(SatPuzzleSudokuGeneral):
    '''
    5 standard sudokus overlapping in a '+' shape
    '''
    def __init__(self, givens: List[List[int]]):
        # the corners are ignored
        assert len(givens) == 15 and all(len(row) == 15 for row in givens)
        areas = []
        for r in chain(range(3),range(12,15)): # top/bottom rows
            areas.append(list(range(15*r+3,15*r+12)))
        for c in chain(range(3),range(12,15)): # left/right columns
            areas.append(list(range(15*3+c,15*12+c,15)))
        for r in range(3,12): # middle rows
            areas.append(list(range(15*r+0,15*r+9)))
            areas.append(list(range(15*r+3,15*r+12)))
            areas.append(list(range(15*r+6,15*r+15)))
        for c in range(3,12): # middle columns
            areas.append(list(range(15*0+c,15*9+c,15)))
            areas.append(list(range(15*3+c,15*12+c,15)))
            areas.append(list(range(15*6+c,15*15+c,15)))
        # set all 3x3 blocks
        for r in range(0,15,3):
            for c in range(0,15,3):
                areas.append([15*(r+rr)+(c+cc) for rr in range(3) for cc in range(3)])
        givens2 = [row[:] for row in givens]
        # set corners to 1..9 because they are irrelevant
        for rr in range(3):
            for cc in range(3):
                n = 3*rr+cc+1 # (0..8) + 1
                givens2[rr][cc] = givens2[12+rr][cc] = givens2[rr][12+cc] = givens2[12+rr][12+cc] = n
        super().__init__(15*15,9,areas,sum(givens2,[]))
    def toSol(self, satSol: List[int]) -> List[List[int]]:
        superSol = super().toSol(satSol)
        sol = [superSol[i:i+15] for i in range(0,15*15,15)]
        # set corners to zeroes and return
        for rr in range(3):
            for cc in range(3):
                sol[rr][cc] = sol[12+rr][cc] = sol[rr][12+cc] = sol[12+rr][12+cc] = 0
        return sol

class SatPuzzleSudokuGattai8(SatPuzzleSudokuGeneral):
    '''
    8 standard overlapping sudokus in a particular arrangement
    '''
    def __init__(self, givens: List[List[int]]):
        assert len(givens) == 21 and all(len(row) == 33 for row in givens)
        areas = []
        for r in chain(range(0,9),range(12,21)): # outer grid rows
            areas.append(list(range(33*r+0,33*r+9)))
            areas.append(list(range(33*r+12,33*r+21)))
            areas.append(list(range(33*r+24,33*r+33)))
        for r in range(6,15): # inner grid rows
            areas.append(list(range(33*r+6,33*r+15)))
            areas.append(list(range(33*r+18,33*r+27)))
        for c in chain(range(0,9),chain(range(12,21),range(24,33))): # outer grid cols
            areas.append(list(range(33*0+c,33*9+c,33)))
            areas.append(list(range(33*12+c,33*21+c,33)))
        for c in chain(range(6,15),range(18,27)): # inner grid cols
            areas.append(list(range(33*6+c,33*15+c,33)))
        for r in range(0,21,3): # all 3x3 blocks
            for c in range(0,33,3):
                areas.append([33*(r+rr)+(c+cc) for rr in range(3) for cc in range(3)])
        givens2 = [row[:] for row in givens]
        # set blank spaces to 1..9
        blank_corners = [(0,9),(0,21),(3,9),(3,21),(9,0),(9,3),(9,15),(9,27),(9,30),(15,9),(15,21),(18,9),(18,21)]
        for rr in range(3):
            for cc in range(3):
                n = 3*rr+cc+1
                for r,c in blank_corners:
                    givens2[r+rr][c+cc] = n
        super().__init__(21*33,9,areas,sum(givens2,[]))
    def toSol(self, satSol: List[int]) -> List[List[int]]:
        superSol = super().toSol(satSol)
        sol = [superSol[i:i+33] for i in range(0,21*33,33)]
        blank_corners = [(0,9),(0,21),(3,9),(3,21),(9,0),(9,3),(9,15),(9,27),(9,30),(15,9),(15,21),(18,9),(18,21)]
        for rr in range(3):
            for cc in range(3):
                for r,c in blank_corners:
                    sol[r+rr][c+cc] = 0
        return sol

class SatPuzzleSudokuConsecutive(SatPuzzleLatinSquare):
    '''
    Latin square with markings for orthogonally adjacent cells having consecutive values
    '''
    def __init__(self, size: int, givens: List[List[int]], consecH: List[List[bool]], consecV: List[List[bool]]):
        N = size
        super().__init__(N,givens)
        assert len(consecH) == N-1 and all(len(row) == N for row in consecH)
        assert len(consecV) == N and all(len(row) == N-1 for row in consecV)
        self.consecH = [row[:] for row in consecH]
        self.consecV = [row[:] for row in consecV]
    def toCnf(self) -> List[List[int]]:
        result = super().toCnf()
        N = self.nums
        x = lambda r,c,n : 1 + (r*N+c)*N + (n-1) # get variable number
        consec = [(a,b) for a in range(1,N+1) for b in range(1,N+1) if a == b+1 or a+1 == b]
        nonconsec = [(a,b) for a in range(1,N+1) for b in range(1,N+1) if a != b and a != b+1 and a+1 != b]
        for r in range(N-1): # horizontal markers - (r,c) and (r+1,c)
            for c in range(N):
                # if must be consecutive, then for any pair of nonconsecutive numbers, either cell is not equal to it
                # if must not be consecutive, then for any pair of consecutive numbers, either cell is not equal to it
                for a,b in nonconsec if self.consecH[r][c] else consec:
                    result.append([-x(r,c,a),-x(r+1,c,b)])
        for r in range(N): # vertical markers - (r,c) and (r,c+1)
            for c in range(N-1):
                for a,b in nonconsec if self.consecV[r][c] else consec:
                    result.append([-x(r,c,a),-x(r,c+1,b)])
        return result
