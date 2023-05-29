from collections import deque
import sys
import heapq

from collections import defaultdict
import datetime
import time


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


def beacon(cellIdx, strength):
    return "BEACON " + str(cellIdx) + " " + str(strength)


def line(sourceIdx: int, targetIdx: int, strength: int) -> str:
    return "LINE " + str(sourceIdx) + " " + str(targetIdx) + " " + str(strength)


def msg(txt) -> str:
    return "MESSAGE " + txt


def debug(txt, indent=0):
    print("  " * indent, end="", file=sys.stderr, flush=True)
    print(txt, file=sys.stderr, flush=True)


def print_value(_str, indent=0):
    if type(eval(_str)) is list:
        debug("{}:(len:{}) {}".format(_str, len(eval(_str)), eval(_str)), indent=indent)
    else:
        debug("{}: {}".format(_str, eval(_str)), indent=indent)


def print_game_phase():
    debug(f"game_phase: {game_phase} ({game_phase_dict[game_phase]})")


# start input
cells: list[Cell] = []
number_of_cells = int(input())  # amount of hexagonal cells in this map

# 時間計測開始
time_sta = time.perf_counter()


INF = number_of_cells + 1
# distance matrix
dist = []
for _ in range(number_of_cells):
    dist.append([INF for i in range(number_of_cells)])

for i in range(number_of_cells):
    dist[i][i] = 0

initial_resources_total: int = 0
init_crystal_list: list[int] = []
init_egg_list: list[int] = []
init_crystal_total: int = 0
init_egg_total: int = 0

for i in range(number_of_cells):
    inputs = [int(j) for j in input().split()]
    cell_type = inputs[0]  # 0 for empty, 1 for eggs, 2 for crystal
    if cell_type == 2:
        init_crystal_list.append(i)
    if cell_type == 1:
        init_egg_list.append(i)
    # the initial amount of eggs/crystals on this cell
    initial_resources = inputs[1]
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

initial_resources_total = init_crystal_total + init_egg_total
number_of_bases: int = int(input())
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
        for j in range(i, number_of_cells):
            dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
            dist[j][i] = dist[i][j]


acc = 0
visit_crystal_list: list[int] = []
middle_crystal_list: list[int] = []
my_close_crystal_list: list[int] = []
for i in sorted(init_crystal_list, key=lambda idx: (dist[my_bases[0]][idx])):
    index = cells[i].index
    if abs(dist[my_bases[0]][index] - dist[opp_bases[0]][index]) <= 2:
        middle_crystal_list.append(i)
    if dist[my_bases[0]][index] * 3 < dist[opp_bases[0]][index]:
        my_close_crystal_list.append(i)
    acc += cells[i].resources
    if init_crystal_total * 0 <= acc * 100 <= init_crystal_total * 60:
        visit_crystal_list.append(i)


# TODO: base複数対応
visit_egg_list: list[int] = list(
    filter(
        lambda idx: dist[my_bases[0]][idx] < dist[opp_bases[0]][idx] * 1.5,
        init_egg_list,
    )
)

# TODO:attackモードを追加。残量の多いmiddle crystalに全力投下して奪いに行く
# Game Phasing Indicator
game_phase_dict = {
    0: "Early Game",
    1: "Mid Game",
    2: "Late Game",
    10: "only one crystal",
    11: "few resources",
    12: "attack middle crystal",
}
game_phase = 0
if len(my_bases) > 1:
    game_phase = 1

# 時間計測終了
time_end = time.perf_counter()
# 経過時間（秒）
debug("init time[ms]:" + str((time_end - time_sta) * 1000))


# TODO: ループの所要時間を計測して、出力する→最終版では削除する
# game loop
while True:
    # 時間計測開始
    time_sta = time.perf_counter()
    # Input Game State
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

        my_ants_total += my_ants

        if resources > 0:
            if cells[i].cell_type == 1:
                egg_list.append(i)
                egg_resource_total += resources

            if cells[i].cell_type == 2:
                crystal_list.append(i)
                crystal_resource_total += resources

    # A. Pre processing
    resources_total = egg_resource_total + crystal_resource_total
    progress_indicator = 1 - (resources_total / initial_resources_total)
    debug(f"progress_indicator: {progress_indicator}")

    # game phase 10: only one crystal
    if len(crystal_list) == 1 and my_ants_total > cells[crystal_list[0]].resources:
        game_phase = 10

    # game phase 11: few resources
    if my_ants_total * 2 > crystal_resource_total:
        game_phase = 11

    # B. Game strategy
    actions = []
    #  if nearest is egg, go to egg
    if game_phase == 0:
        # TODO: baseの複数対応
        print_game_phase()
        nearest_resource_list = sorted(
            crystal_list + egg_list,
            key=lambda idx: (dist[my_bases[0]][idx], cells[idx].cell_type),
        )
        nearest_resource_idx = nearest_resource_list[0]
        if not (
            cells[nearest_resource_idx].cell_type == 1
            and cells[nearest_resource_idx].resources > 0
            and progress_indicator < 0.3
        ):
            game_phase = 1
        else:
            nearest_resources_amount = 0
            nearest_resource_path_way_long = 0
            for i in range(len(nearest_resource_list)):
                if (
                    cells[nearest_resource_list[i]].cell_type == 1
                    and dist[my_bases[0]][nearest_resource_list[i]]
                    == dist[my_bases[0]][nearest_resource_list[0]]
                ):
                    actions.append(line(my_bases[0], nearest_resource_list[i], 1))
                    nearest_resources_amount += cells[
                        nearest_resource_list[i]
                    ].resources
                    nearest_resource_path_way_long += dist[my_bases[0]][
                        nearest_resource_list[i]
                    ]
                    for neighbor in cells[nearest_resource_list[i]].neighbors:
                        if cells[neighbor].cell_type == 1:
                            actions.append(beacon(neighbor, 1))
                            nearest_resources_amount += cells[neighbor].resources
                            nearest_resource_path_way_long += 1
                else:
                    break
            # my ants are enough to get all eggs
            if (
                nearest_resources_amount * nearest_resource_path_way_long
                > my_ants_total
            ):
                print(";".join(actions))
                continue
            else:
                debug("TARGET CHANGE: enough to egg", 2)
                game_phase = 1

    # game phase 11: few resources
    if game_phase == 11:
        print_game_phase()
        visit_egg_list = []
        game_phase = 1

    # main strategy
    if game_phase == 1:
        print_game_phase()
        uf = UnionFind(number_of_cells)
        target_crystal_list = visit_crystal_list.copy()
        if (
            len(list(filter(lambda idx: cells[idx].resources > 0, middle_crystal_list)))
            > 0
        ):
            debug("TARGET CHANGE: middle_crystal", 2)
            print_value("middle_crystal_list", 2)
            target_crystal_list = middle_crystal_list.copy()

        visit_resource_list: list[int] = list(
            filter(
                lambda idx: cells[idx].resources > 0
                and min(map(lambda base: dist[base][idx], my_bases)) * 1.5
                < my_ants_total,
                target_crystal_list + visit_egg_list,
            )
        )
        if len(set(visit_resource_list) - set(my_close_crystal_list)) > 0:
            debug("TARGET CHANGE: close_crystal", 2)
            visit_resource_list = list(
                set(visit_resource_list) - set(my_close_crystal_list)
            )
        print_value("target_crystal_list", 2)
        print_value("visit_resource_list", 2)
        rest_budget = my_ants_total
        visit_resource_list.sort(key=lambda idx: dist[my_bases[0]][idx])

        connected_to_base = my_bases.copy()
        que = deque()
        for base in my_bases:
            que.append(base)
        for resource in visit_resource_list:
            que.append(resource)

        history_dict = {}
        while len(que) > 0:
            debug(f"----new loop---", 1)
            print_value("connected_to_base", 2)
            print_value("que", 2)
            print_value("my_ants_total", 2)
            print_value("len(connected_to_base)", 2)
            rest_budget = my_ants_total - len(connected_to_base)
            print_value("rest_budget", 2)
            current_pos_idx: int = que.popleft()

            if current_pos_idx in connected_to_base:
                continue
            nearest_path = sorted(
                list(
                    filter(
                        lambda x: not uf.same(x, current_pos_idx),
                        connected_to_base,
                    )
                ),
                key=lambda x: (
                    dist[current_pos_idx][x],
                    dist[my_bases[0]][x],
                    cells[x].my_ants * -1,
                ),
            )[0]
            next_neighbors_list = list(
                filter(
                    lambda x: dist[nearest_path][x] + 1
                    == dist[nearest_path][current_pos_idx],
                    cells[current_pos_idx].neighbors,
                )
            )
            next_neighbors_list.sort(
                key=lambda x: (dist[my_bases[0]][x], -1 * cells[x].my_ants),
                reverse=False,
            )
            next_cell = next_neighbors_list[0]

            print_value("nearest_path", 2)
            print_value("next_neighbors_list", 2)
            print_value("dist[current_pos_idx][nearest_path]", 2)

            # ant resource check: ants are not enough to get the resource, discard
            if rest_budget < dist[current_pos_idx][nearest_path]:
                continue

            # infinite-loop check
            # force to go the first next cell
            if history_dict.get(current_pos_idx, -1) == len(que):
                uf.union(current_pos_idx, next_cell)
                connected_to_base.append(current_pos_idx)
                que.appendleft(next_cell)
                continue
            history_dict[current_pos_idx] = len(que)

            # path defined
            if len(next_neighbors_list) == 1:
                uf.union(current_pos_idx, next_cell)
                connected_to_base.append(current_pos_idx)
                que.appendleft(next_cell)
                continue

            # path not defined
            # one of the next_cells is already connected to base
            for one_of_next_cell in next_neighbors_list:
                if one_of_next_cell in connected_to_base:
                    uf.union(current_pos_idx, one_of_next_cell)
                    continue

            # don't handle, push back to que to retry
            que.append(current_pos_idx)
        debug(f"===loop end===", 1)
        print_value("len(connected_to_base)", 2)
        print_value("connected_to_base", 2)

        # start to union, connect isolated islands to base
        connected_to_base.sort(key=lambda idx: dist[my_bases[0]][idx])
        verified_connection_cells = my_bases.copy()
        # TODO: baseの複数対応
        for current_pos_idx in connected_to_base:
            if not any(map(lambda x: uf.same(x, current_pos_idx), my_bases)):
                nearest_path = sorted(
                    verified_connection_cells,
                    key=lambda x: (dist[current_pos_idx][x], cells[x].my_ants * -1),
                )[0]
                actions.append(line(nearest_path, current_pos_idx, 1))
                uf.union(nearest_path, current_pos_idx)
            verified_connection_cells.append(current_pos_idx)

        print_value("len(connected_to_base)", 2)
        print_value("connected_to_base", 2)

        for cell in connected_to_base:
            if cell in my_bases:
                if any(map(lambda x: x in connected_to_base, cells[cell].neighbors)):
                    actions.append(beacon(cell, 1))
            else:
                actions.append(beacon(cell, 1))

    if game_phase == 10:
        print_game_phase()
        last_crystal_idx = crystal_list[0]
        nearest_base = sorted(
            my_bases,
            key=lambda idx: (dist[last_crystal_idx][idx], cells[idx].my_ants * -1),
        )[0]
        actions.append(line(last_crystal_idx, nearest_base, 1))

    if len(actions) == 0:
        for i in sorted(init_crystal_list, key=lambda idx: (dist[my_bases[0]][idx])):
            if cells[i].resources > 0:
                actions.append(line(my_bases[0], i, 32))
    print(";".join(actions))

    # 時間計測終了
    time_end = time.perf_counter()
    # 経過時間（秒）
    debug("loop time[ms]:" + str((time_end - time_sta) * 1000))
