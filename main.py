from collections import deque


class Cell(object):
    index: int
    cell_type: int
    resources: int
    neighbors: list[int]
    my_ants: int
    opp_ants: int
    dist_from_base: int

    def __init__(
        self,
        index: int,
        cell_type: int,
        resources: int,
        neighbors: list[int],
        my_ants: int,
        opp_ants: int,
        dist_from_base: int,
    ):
        self.index = index
        self.cell_type = cell_type
        self.resources = resources
        self.neighbors = neighbors
        self.my_ants = my_ants
        self.opp_ants = opp_ants
        self.dist_from_base = dist_from_base


def line(sourceIdx, targetIdx, strength) -> str:
    return "LINE " + str(sourceIdx) + " " + str(targetIdx) + " " + str(strength)


def msg(txt) -> str:
    return "MESSAGE " + txt


cells: list[Cell] = []
egg_cell_idx = -1
crystal_cell_idx = -1
number_of_cells = int(input())  # amount of hexagonal cells in this map

dist = []
for _ in range(number_of_cells):
    dist.append([-1 for i in range(number_of_cells)])

for i in range(number_of_cells):
    dist[i][i] = 0

for i in range(number_of_cells):
    inputs = [int(j) for j in input().split()]
    cell_type = inputs[0]  # 0 for empty, 1 for eggs, 2 for crystal
    if cell_type == 2 and crystal_cell_idx == -1:
        crystal_cell_idx = i
    if cell_type == 1:
        egg_cell_idx = i
    # the initial amount of eggs/crystals on this cell
    initial_resources = inputs[1]
    # the index of the neighbouring cell for each direction
    neigh_0 = inputs[2]
    neigh_1 = inputs[3]
    neigh_2 = inputs[4]
    neigh_3 = inputs[5]
    neigh_4 = inputs[6]
    neigh_5 = inputs[7]
    cell: Cell = Cell(
        index=i,
        cell_type=cell_type,
        resources=initial_resources,
        neighbors=list(
            filter(
                lambda id: id > -1,
                [neigh_0, neigh_1, neigh_2, neigh_3, neigh_4, neigh_5],
            )
        ),
        my_ants=0,
        opp_ants=0,
        dist_from_base=-1,
    )
    cells.append(cell)
number_of_bases = int(input())
my_bases: list[int] = []
for i in input().split():
    my_base_index = int(i)
    my_bases.append(my_base_index)
opp_bases: list[int] = []
for i in input().split():
    opp_base_index = int(i)
    opp_bases.append(opp_base_index)

# preset dist_from_base
q = deque()
q.append(my_bases[0])
cells[my_bases[0]].dist_from_base = 0

while len(q) > 0:
    target_cell_idx = q.popleft()

    current_dist = cells[target_cell_idx].dist_from_base
    for i in range(len(cells[target_cell_idx].neighbors)):
        next_idx = cells[target_cell_idx].neighbors[i]
        if cells[next_idx].dist_from_base == -1:
            cells[next_idx].dist_from_base = current_dist + 1
            q.append(next_idx)


# game loop
nearest_cell_idx = -1
nearest_egg_idx = -1
while True:
    egg_list = []
    crystal_list = []

    for i in range(number_of_cells):
        inputs = [int(j) for j in input().split()]
        resources = inputs[0]  # the current amount of eggs/crystals on this cell
        my_ants = inputs[1]  # the amount of your ants on this cell
        opp_ants = inputs[2]  # the amount of opponent ants on this cell

        cells[i].resources = resources
        cells[i].my_ants = my_ants
        cells[i].opp_ants = opp_ants

        if cells[i].cell_type == 1 and resources > 0:
            egg_list.append(i)

        if cells[i].cell_type == 2 and resources > 0:
            crystal_list.append(i)

    egg_list.sort(key=lambda idx: cells[idx].dist_from_base)
    crystal_list.sort(key=lambda idx: cells[idx].dist_from_base)

    # WAIT | LINE <sourceIdx> <targetIdx> <strength> | BEACON <cellIdx> <strength> | MESSAGE <text>
    actions = []
    for i in range(0, min(len(crystal_list), 1)):
        if (
            cells[egg_list[0]].dist_from_base
            < cells[crystal_list[i]].dist_from_base * 2
        ):
            actions.append(line(my_bases[0], egg_list[i], 25))

    for i in range(1, min(len(crystal_list), 3)):
        if (
            cells[egg_list[i]].dist_from_base
            < cells[crystal_list[0]].dist_from_base * 2
        ):
            actions.append(line(my_bases[0], egg_list[i], 1))

    actions.append(line(my_bases[0], crystal_list[0], 20))
    for i in range(1, min(len(crystal_list), 5)):
        actions.append(line(my_bases[0], crystal_list[i], max(0, 10 - i * 2)))

    # To debug: print("Debug messages...", file=sys.stderr, flush=True)
    if len(actions) == 0:
        print("WAIT")
    else:
        print(";".join(actions))
