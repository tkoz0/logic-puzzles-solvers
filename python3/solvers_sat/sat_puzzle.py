from itertools import chain
import pycosat
from typing import Any, Generator, Iterator, List, Set, Tuple

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
    def __init__(self, givens: List[List[int]]):
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
    Latin square with diagonals
    '''
    def __init__(self, givens: List[List[int]]):
        N = len(givens)
        super().__init__(givens)
        self.areas.append(list(range(0,N*N,N+1)))
        self.areas.append(list(range(N-1,N*N-1,N-1)))

class SatPuzzleSudokuStandard(SatPuzzleLatinSquare):
    '''
    Extends latin square by adding rectangular blocks
    '''
    def __init__(self, blockR: int, blockC: int, givens: List[List[int]]):
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
    def __init__(self, givens: List[List[int]], areas: List[List[int]]):
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
        super().__init__(givens,areas)
        N = len(givens)
        self.areas.append(list(range(0,N*N,N+1)))
        self.areas.append(list(range(N-1,N*N-1,N-1)))

# TODO make a class for the overlapping sudoku types generalized with blockR,blockC using grid aligned puzzles sharing some blocks

class SatPuzzleSudokuOverlap(SatPuzzleSudokuGeneral):
    '''
    Generalization of sudoku variants that overlap standard puzzles
    - specify list of top left corners using block coordinates
    '''
    def __init__(self, blockR: int, blockC: int, givens: List[List[int]], corners: List[Tuple[int,int]]):
        N = blockR*blockC
        brows = max(p[0] for p in corners) + blockC
        bcols = max(p[1] for p in corners) + blockR
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
            assert pr % blockR == 0 and pc % blockC == 0
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
    - consecutive pairs are specified as a set of coordinate pairs which must only be orthogonally adjacent
    '''
    def __init__(self, givens: List[List[int]], consec_pairs: Set[Tuple[Tuple[int,int],Tuple[int,int]]]):
        N = len(givens)
        super().__init__(givens)
        assert all(0 <= r1 < N and 0 <= c1 < N and 0 <= r2 < N and 0 <= c2 < N for (r1,c1),(r2,c2) in consec_pairs)
        self.consec_pairs = set(p for p in consec_pairs)
    def toCnf(self) -> List[List[int]]: # extend to add extra constraints
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
    Blank standard sudoku grid with
    - white circle means adjacent values must be consecutive
    - black circle means one of the value is double of the other
    - adjacent cells without a circle must be neither
    '''
    def __init__(self, blockR: int, blockC, white: Set[Tuple[Tuple[int,int],Tuple[int,int]]], black: Set[Tuple[Tuple[int,int],Tuple[int,int]]]):
        N = blockR*blockC
        super().__init__(blockR,blockC,[[0]*N for _ in range(N)])
        assert all(0 <= r1 < N and 0 <= c1 < N and 0 <= r2 < N and 0 <= c2 < N for (r1,c1),(r2,c2) in white)
        assert all(0 <= r1 < N and 0 <= c1 < N and 0 <= r2 < N and 0 <= c2 < N for (r1,c1),(r2,c2) in black)
        assert white & black == set()
        self.white = set(p for p in white)
        self.black = set(p for p in black)
    def toCnf(self) -> List[List[int]]: # extend to add extra constraints
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
    Variant where lines between cells indicate that the 2 must sum to the given magic number
    '''
    def __init__(self, givens: List[List[int]], magic: int, pairs: Set[Tuple[Tuple[int,int],Tuple[int,int]]]):
        N = len(givens)
        super().__init__(givens)
        self.magic = magic
        assert all(0 <= r1 < N and 0 <= c1 < N and 0 <= r2 < N and 0 <= c2 < N for (r1,c1),(r2,c2) in pairs)
        self.pairs = set(p for p in pairs)
    def toCnf(self) -> List[List[int]]:
        result = super().toCnf()
        N = self.nums
        x = lambda r,c,n : 1 + (r*N+c)*N + (n-1) # get variable number
        not_magic_sum = [(a,b) for a in range(1,N+1) for b in range(1,N+1) if a != b and a+b != self.magic]
        for (r1,c1),(r2,c2) in self.pairs:
            for a,b in not_magic_sum: # at least 1 of these must not be set
                result.append([-x(r1,c1,a),-x(r2,c2,b)])
        return result

class SatPuzzleSudokuSamurai(SatPuzzleSudokuGeneral):
    '''
    5 standard sudokus overlapping in a '+' shape
    '''
    def __init__(self, givens: List[List[int]]):
        # some parts of the area are ignored
        assert len(givens) == 21 and all(len(row) == 21 for row in givens)
        areas = []
        for r in chain(range(0,9),range(12,21)): # top/bottom rows
            areas.append(list(range(21*r+0,21*r+9)))
            areas.append(list(range(21*r+12,21*r+21)))
        for c in chain(range(0,9),range(12,21)): # left/right columns
            areas.append(list(range(21*0+c,21*9+c,21)))
            areas.append(list(range(21*12+c,21*21+c,21)))
        for r in range(6,15): # middle rows
            areas.append(list(range(21*r+6,21*r+15)))
        for c in range(6,15): # middle columns
            areas.append(list(range(21*6+c,21*15+c,21)))
        for r in range(0,21,3): # all 3x3 blocks
            for c in range(0,21,3):
                areas.append([21*(r+rr)+(c+cc) for rr in range(3) for cc in range(3)])
        givens2 = [row[:] for row in givens]
        # set ignored parts to 1..9
        for rr in range(3):
            for cc in range(3):
                n = 3*rr+cc+1
                givens2[rr][9+cc] = givens2[3+rr][9+cc] = givens2[9+rr][cc] = givens2[9+rr][3+cc] = n
                givens2[9+rr][15+cc] = givens2[9+rr][18+cc] = givens2[15+rr][9+cc] = givens2[18+rr][9+cc] = n
        super().__init__(21*21,9,areas,sum(givens2,[]))
    def toSol(self, satSol: List[int]) -> List[List[int]]:
        superSol = super().toSol(satSol)
        sol = [superSol[i:i+21] for i in range(0,21*21,21)]
        # set corners to zeroes and return
        for rr in range(3):
            for cc in range(3):
                sol[rr][9+cc] = sol[3+rr][9+cc] = sol[9+rr][cc] = sol[9+rr][3+cc] = 0
                sol[9+rr][15+cc] = sol[9+rr][18+cc] = sol[15+rr][9+cc] = sol[18+rr][9+cc] = 0
        return sol

class SatPuzzleSudokuOddEven(SatPuzzleSudokuStandard):
    '''
    Variant with all odd numbers marked by circles
    '''
    def __init__(self, blockR: int, blockC: int, givens: List[List[int]], odds: List[List[bool]]):
        super().__init__(blockR,blockC,givens)
        N = len(givens)
        assert len(odds) == N and all(len(row) == N for row in odds)
        self.odds = [row[:] for row in odds]
    def toCnf(self) -> List[List[int]]:
        result = super().toCnf()
        N = self.nums
        x = lambda r,c,n : 1 + (r*N+c)*N + (n-1) # get variable number
        for r,row in enumerate(self.odds):
            for c,odd in enumerate(row):
                if odd:
                    for n in range(2,N+1,2):
                        result.append([-x(r,c,n)])
        return result

class SatPuzzleSudokuOddEvenSamurai(SatPuzzleSudokuSamurai):
    def __init__(self, givens: List[List[int]], odds: List[List[bool]]):
        super().__init__(givens)
        assert len(odds) == 21 and all(len(row) == 21 for row in odds)
        self.odds = [row[:] for row in odds]
        # have to set special odds for the unused parts of the samurai shape
        for n in range(1,10,2):
            rr,cc = divmod(n-1,3)
            self.odds[rr][9+cc] = self.odds[3+rr][9+cc] = self.odds[9+rr][cc] = self.odds[9+rr][3+cc] = True
            self.odds[9+rr][15+cc] = self.odds[9+rr][18+cc] = self.odds[15+rr][9+cc] = self.odds[18+rr][9+cc] = True
    def toCnf(self) -> List[List[int]]:
        result = super().toCnf()
        x = lambda r,c,n : 1 + (r*21+c)*9 + (n-1)
        for r,row in enumerate(self.odds):
            for c,odd in enumerate(row):
                if odd: # cell must be odd
                    result.append([x(r,c,n) for n in range(1,10,2)])
                else: # cell must be even
                    result.append([x(r,c,n) for n in range(2,10,2)])
        return result

class SatPuzzleSudokuMarginalSum(SatPuzzleSudokuStandard):
    '''
    Sudoku with no given cells. The clues are the sum of the first 3 numbers
    from the edges. This class extends that notion to other block sizes.
    '''
    def __init__(self, blockR: int, blockC: int, top: List[int], bottom: List[int], left: List[int], right: List[int]):
        N = blockR*blockC
        super().__init__(blockR,blockC,[[0]*N for _ in range(N)])
        assert len(top) == len(bottom) == len(left) == len(right) == N
        self.top = top[:]
        self.bottom = bottom[:]
        self.left = left[:]
        self.right = right[:]
    def toCnf(self) -> List[List[int]]:
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

class SatPuzzleSudokuShogun(SatPuzzleSudokuGeneral):
    '''
    11 standard overlapping sudokus in a particular arrangement
    '''
    def __init__(self, givens: List[List[int]]):
        assert len(givens) == 21 and all(len(row) == 45 for row in givens)
        areas = []
        for r in chain(range(0,9),range(12,21)): # outer grid rows
            areas.append(list(range(45*r+0,45*r+9)))
            areas.append(list(range(45*r+12,45*r+21)))
            areas.append(list(range(45*r+24,45*r+33)))
            areas.append(list(range(45*r+36,45*r+45)))
        for r in range(6,15): # inner grid rows
            areas.append(list(range(45*r+6,45*r+15)))
            areas.append(list(range(45*r+18,45*r+27)))
            areas.append(list(range(45*r+30,45*r+39)))
        for c in chain(range(0,9),chain(range(12,21),chain(range(24,33),range(36,45)))): # outer grid cols
            areas.append(list(range(45*0+c,45*9+c,45)))
            areas.append(list(range(45*12+c,45*21+c,45)))
        for c in chain(range(6,15),chain(range(18,27),range(30,39))): # inner grid cols
            areas.append(list(range(45*6+c,45*15+c,45)))
        for r in range(0,21,3): # all 3x3 blocks
            for c in range(0,45,3):
                areas.append([45*(r+rr)+(c+cc) for rr in range(3) for cc in range(3)])
        givens2 = [row[:] for row in givens]
        # set blank spaces to 1..9
        blank_corners = [(0,9),(0,21),(0,33),(3,9),(3,21),(3,33),(9,0),(9,3),(9,15),(9,27),(9,39),(9,42),(15,9),(15,21),(15,33),(18,9),(18,21),(18,33)]
        for rr in range(3):
            for cc in range(3):
                n = 3*rr+cc+1
                for r,c in blank_corners:
                    givens2[r+rr][c+cc] = n
        super().__init__(21*45,9,areas,sum(givens2,[]))
    def toSol(self, satSol: List[int]) -> List[List[int]]:
        superSol = super().toSol(satSol)
        sol = [superSol[i:i+45] for i in range(0,21*45,45)]
        blank_corners = [(0,9),(0,21),(0,33),(3,9),(3,21),(3,33),(9,0),(9,3),(9,15),(9,27),(9,39),(9,42),(15,9),(15,21),(15,33),(18,9),(18,21),(18,33)]
        for rr in range(3):
            for cc in range(3):
                for r,c in blank_corners:
                    sol[r+rr][c+cc] = 0
        return sol

class SatPuzzleSudokuSohei(SatPuzzleSudokuGeneral):
    '''
    4 standard overlapping sudokus in a particular arrangement
    '''
    def __init__(self, givens: List[List[int]]):
        assert len(givens) == 21 and all(len(row) == 21 for row in givens)
        areas = []
        for r in chain(range(0,9),range(12,21)): # top/bottom rows
            areas.append(list(range(21*r+6,21*r+15)))
        for r in range(6,15): # left/right rows
            areas.append(list(range(21*r+0,21*r+9)))
            areas.append(list(range(21*r+12,21*r+21)))
        for c in chain(range(0,9),range(12,21)): # left/right rolumns
            areas.append(list(range(21*6+c,21*15+c,21)))
        for c in range(6,15): # top/bottom columns
            areas.append(list(range(21*0+c,21*9+c,21)))
            areas.append(list(range(21*12+c,21*21+c,21)))
        for r in range(0,21,3): # 3x3 blocks
            for c in range(0,21,3):
                areas.append([21*(r+rr)+(c+cc) for rr in range(3) for cc in range(3)])
        givens2 = [row[:] for row in givens]
        # set blank spaces to 1..9
        blank_corners = [(0,0),(0,3),(0,15),(0,18),(3,0),(3,3),(3,15),(3,18),(9,9),(15,0),(15,3),(15,15),(15,18),(18,0),(18,3),(18,15),(18,18)]
        for rr in range(3):
            for cc in range(3):
                n = 3*rr+cc+1
                for r,c in blank_corners:
                    givens2[r+rr][c+cc] = n
        super().__init__(21*21,9,areas,sum(givens2,[]))
    def toSol(self, satSol: List[int]) -> List[List[int]]:
        superSol = super().toSol(satSol)
        sol = [superSol[i:i+21] for i in range(0,21*21,21)]
        blank_corners = [(0,0),(0,3),(0,15),(0,18),(3,0),(3,3),(3,15),(3,18),(9,9),(15,0),(15,3),(15,15),(15,18),(18,0),(18,3),(18,15),(18,18)]
        for rr in range(3):
            for cc in range(3):
                for r,c in blank_corners:
                    sol[r+rr][c+cc] = 0
        return sol
