from collections import deque
import sys
import heapq

from collections import defaultdict


class UnionFind:
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n

    def find(self, x):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.parents[x] > self.parents[y]:
            x, y = y, x

        self.parents[x] += self.parents[y]
        self.parents[y] = x

    def size(self, x):
        return -self.parents[self.find(x)]

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def members(self, x):
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]

    def roots(self):
        return [i for i, x in enumerate(self.parents) if x < 0]

    def group_count(self):
        return len(self.roots())

    def all_group_members(self):
        group_members = defaultdict(list)
        for member in range(self.n):
            group_members[self.find(member)].append(member)
        return group_members

    def __str__(self):
        return "\n".join(f"{r}: {m}" for r, m in self.all_group_members().items())


class Cell(object):
    index: int
    cell_type: int
    resources: int
    neighbors: list[int]
    my_ants: int
    opp_ants: int
    dist_from_base: int
    is_middle_point_between_us: bool

    def __init__(
        self,
        index: int,
        cell_type: int,
        resources: int,
        neighbors: list[int],
        my_ants: int,
        opp_ants: int,
        dist_from_base: int,
        is_middle_point_between_us: bool = False,
    ):
        self.index = index
        self.cell_type = cell_type
        self.resources = resources
        self.neighbors = neighbors
        self.my_ants = my_ants
        self.opp_ants = opp_ants
        self.dist_from_base = dist_from_base
        self.is_middle_point_between_us = is_middle_point_between_us


def beacon(cellIdx, strength):
    return "BEACON " + str(cellIdx) + " " + str(strength)


def line(sourceIdx: int, targetIdx: int, strength: int) -> str:
    return "LINE " + str(sourceIdx) + " " + str(targetIdx) + " " + str(strength)


def msg(txt) -> str:
    return "MESSAGE " + txt


def debug(txt):
    print(txt, file=sys.stderr, flush=True)


cells: list[Cell] = []
number_of_cells = int(input())  # amount of hexagonal cells in this map

INF = number_of_cells + 1
# distance matrix
dist = []
for _ in range(number_of_cells):
    dist.append([INF for i in range(number_of_cells)])

for i in range(number_of_cells):
    dist[i][i] = 0

initial_resources_total = 0
init_crystal_list: list[int] = []
init_egg_list: list[int] = []
init_crystal_total = 0
init_egg_total = 0

for i in range(number_of_cells):
    inputs = [int(j) for j in input().split()]
    cell_type = inputs[0]  # 0 for empty, 1 for eggs, 2 for crystal
    if cell_type == 2:
        init_crystal_list.append(i)
    if cell_type == 1:
        init_egg_list.append(i)
    # the initial amount of eggs/crystals on this cell
    initial_resources = inputs[1]
    initial_resources_total += initial_resources
    init_crystal_total += initial_resources if cell_type == 2 else 0
    init_egg_total += initial_resources if cell_type == 1 else 0
    # the index of the neighbouring cell for each direction
    neigh_0 = inputs[2]
    neigh_1 = inputs[3]
    neigh_2 = inputs[4]
    neigh_3 = inputs[5]
    neigh_4 = inputs[6]
    neigh_5 = inputs[7]
    c: Cell = Cell(
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
    cells.append(c)
    for next in c.neighbors:
        dist[i][next] = 1

number_of_bases = int(input())
my_bases: list[int] = []
for i in input().split():
    my_base_index = int(i)
    my_bases.append(my_base_index)
opp_bases: list[int] = []
for i in input().split():
    opp_base_index = int(i)
    opp_bases.append(opp_base_index)


for k in range(number_of_cells):
    for i in range(number_of_cells):
        for j in range(number_of_cells):
            dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
            dist[j][i] = min(dist[i][j], dist[i][k] + dist[k][j])

for c in cells:
    c.is_middle_point_between_us = (
        dist[c.index][my_bases[0]] == dist[c.index][opp_bases[0]]
    )


acc = 0
visit_crystal_list: list[int] = []
for i in sorted(init_crystal_list, key=lambda idx: (dist[my_bases[0]][idx])):
    acc += cells[i].resources
    visit_crystal_list.append(i)
    if acc > init_crystal_total * 90 // 100:
        break


# game loop
while True:
    egg_list: list[int] = []
    crystal_list: list[int] = []
    resources_total = 0
    crystal_resource_total = 0
    egg_resource_total = 0
    my_ants_total = 0
    for i in range(number_of_cells):
        inputs = [int(j) for j in input().split()]
        resources = inputs[0]  # the current amount of eggs/crystals on this cell
        my_ants = inputs[1]  # the amount of your ants on this cell
        opp_ants = inputs[2]  # the amount of opponent ants on this cell

        cells[i].resources = resources
        cells[i].my_ants = my_ants
        cells[i].opp_ants = opp_ants

        resources_total += resources
        my_ants_total += my_ants

        if cells[i].cell_type == 1 and resources > 0:
            egg_list.append(i)
            egg_resource_total += resources

        if cells[i].cell_type == 2 and resources > 0:
            crystal_list.append(i)
            crystal_resource_total += resources

    progress_indicator = (
        initial_resources_total - resources_total
    ) / initial_resources_total

    is_progress_egg_discard = progress_indicator >= 0.5 or crystal_resource_total <= 30

    actions = []

    nearest_resource_list = sorted(
        crystal_list + egg_list,
        key=lambda idx: (dist[my_bases[0]][idx], cells[idx].cell_type),
    )
    nearest_resource_idx = nearest_resource_list[0]

    debug(f"progress_indicator: {progress_indicator}")

    # if nearest is egg, go to egg
    if cells[nearest_resource_idx].cell_type == 1 and progress_indicator < 0.3:
        for i in range(len(nearest_resource_list)):
            if (
                cells[nearest_resource_list[i]].cell_type == 1
                and dist[my_bases[0]][nearest_resource_list[i]]
                == dist[my_bases[0]][nearest_resource_list[0]]
            ):
                actions.append(line(my_bases[0], nearest_resource_list[i], 2))
                for neighbor in cells[nearest_resource_list[i]].neighbors:
                    if cells[neighbor].cell_type == 1:
                        actions.append(beacon(neighbor, 1))
            else:
                break
        print(";".join(actions))
        continue

    # midpoint priority
    if is_progress_egg_discard:
        for c in crystal_list:
            if cells[c].is_middle_point_between_us:
                actions.append(line(my_bases[0], c, 1))
                break
        if len(actions) > 0:
            print(";".join(actions))
            debug("midpoint priority triggered")
            continue

    # main strategy
    uf = UnionFind(number_of_cells)
    visit_resource_list: list[int] = list(
        filter(lambda idx: cells[idx].resources > 0, visit_crystal_list + egg_list)
    )

    visit_resource_list.sort(key=lambda idx: dist[my_bases[0]][idx])

    connected_to_base = [my_bases[0]]
    que = deque()
    que.append(my_bases[0])
    for resource in visit_resource_list:
        que.append(resource)

    resource_with_multi_path_list = []
    history_dict = {}
    while len(que) > 0:
        debug(f"----new loop---")
        debug(f"connected_to_base: {connected_to_base}")
        debug(f"que: {que}")
        current_pos_idx: int = que.popleft()

        if current_pos_idx in connected_to_base:
            continue
        nearest_path = sorted(
            connected_to_base,
            key=lambda x: dist[current_pos_idx][x],
        )[0]
        next_neighbors_list = list(
            filter(
                lambda x: dist[nearest_path][x] + 1
                == dist[nearest_path][current_pos_idx],
                cells[current_pos_idx].neighbors,
            )
        )
        debug(f"nearest_path: {nearest_path}")
        debug(f"next_neighbors_list: {next_neighbors_list}")
        if history_dict[current_pos_idx] == len(que):
            actions.append(line(current_pos_idx, next_neighbors_list[0], 1))
            uf.union(current_pos_idx, next_neighbors_list[0])
            continue
        history_dict[current_pos_idx] = len(que)

        if len(next_neighbors_list) == 1:
            uf.union(current_pos_idx, next_neighbors_list[0])
            connected_to_base.append(current_pos_idx)
            que.appendleft(next_neighbors_list[0])
            continue
        else:
            if next_neighbors_list[0] in connected_to_base:
                uf.union(current_pos_idx, next_neighbors_list[0])
            else:
                que.append(current_pos_idx)

    connected_to_base.sort(key=lambda idx: dist[my_bases[0]][idx])
    verified_connection_cells = [my_bases[0]]
    for current_pos_idx in connected_to_base:
        if not uf.same(my_bases[0], current_pos_idx):
            nearest_path = sorted(
                verified_connection_cells,
                key=lambda x: dist[current_pos_idx][x],
            )[0]
            actions.append(line(nearest_path, current_pos_idx, 1))
        verified_connection_cells.append(current_pos_idx)

    for cell in connected_to_base:
        actions.append(beacon(cell, 16))

    if len(actions) == 0:
        for i in sorted(init_crystal_list, key=lambda idx: (dist[my_bases[0]][idx])):
            if cells[i].resources > 0:
                actions.append(line(my_bases[0], i, 32))
    print(";".join(actions))
