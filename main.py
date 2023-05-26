class Cell(object):
    index: int
    cell_type: int
    resources: int
    neighbors: list[int]
    my_ants: int
    opp_ants: int

    def __init__(
        self,
        index: int,
        cell_type: int,
        resources: int,
        neighbors: list[int],
        my_ants: int,
        opp_ants: int,
    ):
        self.index = index
        self.cell_type = cell_type
        self.resources = resources
        self.neighbors = neighbors
        self.my_ants = my_ants
        self.opp_ants = opp_ants


def line(sourceIdx, targetIdx, strength) -> str:
    return "LINE " + str(sourceIdx) + " " + str(targetIdx) + " " + str(strength)


def msg(txt) -> str:
    return "MESSAGE " + txt


cells: list[Cell] = []
egg_cell_idx = -1
crystal_cell_idx = -1
number_of_cells = int(input())  # amount of hexagonal cells in this map
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

# game loop
target_cell_idx = 0
target_egg_idx = 0
while True:
    cur_cell_idx = -1
    cur_egg_idx = -1
    for i in range(number_of_cells):
        inputs = [int(j) for j in input().split()]
        # the current amount of eggs/crystals on this cell
        resources = inputs[0]
        my_ants = inputs[1]  # the amount of your ants on this cell
        opp_ants = inputs[2]  # the amount of opponent ants on this cell

        cells[i].resources = resources
        cells[i].my_ants = my_ants
        cells[i].opp_ants = opp_ants

        if cells[i].cell_type is 2 and (
            cur_cell_idx is -1 or cells[i].resources > cells[cur_cell_idx].resources
        ):
            cur_cell_idx = i

        if cells[i].cell_type is 1 and (
            cur_egg_idx is -1 or cells[i].resources > cells[cur_egg_idx].resources
        ):
            cur_egg_idx = i

    if cells[target_cell_idx].resources is 0:
        target_cell_idx = cur_cell_idx
    if cells[target_egg_idx].resources is 0:
        target_egg_idx = cur_egg_idx

    # WAIT | LINE <sourceIdx> <targetIdx> <strength> | BEACON <cellIdx> <strength> | MESSAGE <text>
    actions = []
    actions.append(line(my_bases[0], target_cell_idx, 1))
    actions.append(line(my_bases[0], target_egg_idx, 1))

    # To debug: print("Debug messages...", file=sys.stderr, flush=True)
    if len(actions) == 0:
        print("WAIT")
    else:
        print(";".join(actions))
