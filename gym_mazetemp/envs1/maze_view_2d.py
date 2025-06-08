import pygame
import random
import numpy as np
import os


class MazeView2D:
    def __init__(self, maze_name="Maze2D", maze_file_path=None,
                 maze_size=(30, 30), screen_size=(800, 1000),
                 has_loops=False, num_portals=0, render_mode=None):

        self.render_mode = render_mode
        
        # PyGame configurations
        if not pygame.get_init():
            pygame.init()
        
        # Always use a virtual surface
            self.screen = pygame.Surface(screen_size)
        
        # 创建所有必要的surface
        self.background = pygame.Surface(screen_size)
        self.maze_layer = pygame.Surface(screen_size, pygame.SRCALPHA)  # 使用SRCALPHA来支持透明度
        
        # 定义显示区域
        self.maze_area = pygame.Rect(0, 0, 800, 600)  # 上部迷宫区域
        self.plot_area = pygame.Rect(0, 600, 800, 400)  # 下部指标区域
        
        # 初始化surface
        self.background.fill((0, 128, 0), rect=self.maze_area)  # 迷宫区域绿色背景
        self.background.fill((255, 255, 255), rect=self.plot_area)  # 指标区域白色背景
        self.maze_layer.fill((0, 0, 0, 0))  # 透明背景
        
        self.clock = pygame.time.Clock()
        self.__game_over = False
        
        # 初始化指标历史记录
        self.reward_history = []
        self.loss_history = []
        
        # Load a maze
        if maze_file_path is None:
            self.__maze = Maze(maze_size=maze_size, has_loops=has_loops, num_portals=num_portals)
        else:
            if not os.path.exists(maze_file_path):
                dir_path = os.path.dirname(os.path.abspath(__file__))
                rel_path = os.path.join(dir_path, "maze_samples", maze_file_path)
                if os.path.exists(rel_path):
                    maze_file_path = rel_path
                else:
                    raise FileExistsError("Cannot find %s." % maze_file_path)
            self.__maze = Maze(maze_cells=Maze.load_maze(maze_file_path))

        self.maze_size = self.__maze.maze_size
        self.__screen_size = screen_size
        
        # Set the starting point
        self.__entrance = np.zeros(2, dtype=int)

        # Set the Goal
        self.__goal = np.array(self.maze_size) - np.array((1, 1))

        # Create the Robot
        self.__robot = self.entrance
        self.__temp_robot = self.entrance

        # Show the maze
        self.__draw_maze()
        
        # Show the portals
        self.__draw_portals()

        # Show the robot
        self.__draw_robot()

        # Show the entrance
        self.__draw_entrance()

        # Show the goal
        self.__draw_goal()

        ####
        self.__last_best_pos = None
        self.__last_chosen_pos = None

        ########
        self.screen.fill((255, 255, 255), self.plot_area)

    def get_image(self):
        """获取当前画面的图像数据"""
        # 更新画面
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.maze_layer, (0, 0))
        
        # 获取图像数据并转换为RGB格式
        image_data = pygame.surfarray.array3d(self.screen)
        return np.transpose(image_data, (1, 0, 2))  # 返回 (H, W, C) 格式的RGB图像

    def update(self, mode="rgb_array"):
        """更新并返回当前画面"""
        try:
            if self.render_mode is None:
                return None
            
            # 更新迷宫状态
            self.__draw_entrance()
            self.__draw_goal()
            self.__draw_portals()
            self.__draw_robot()
            
            # 更新迷宫区域
            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.maze_layer, (0, 0))
            
            # 返回numpy数组格式的图像
            return self.get_image()
            
        except Exception as e:
            print(f"Error in update: {str(e)}")
            self.__game_over = True
            self.quit_game()
            return None

    def quit_game(self):
        """清理pygame资源"""
        try:
            self.__game_over = True
            pygame.quit()
        except Exception:
            pass

    def move_robot(self, dir, best_action):
        """Move the robot in the specified direction and visualize the best action.
        
        Args:
            dir: The direction to move the robot
            best_action: The best action according to the current policy
        """
        if dir not in self.__maze.COMPASS.keys():
            raise ValueError("dir cannot be %s. The only valid dirs are %s."
                             % (str(dir), str(self.__maze.COMPASS.keys())))

        if self.__maze.is_open(self.temp_robot, dir):
            # update the drawing
            self.__draw_robot(transparency=0)
            self.__robot = self.__temp_robot
            self.__draw_robot(transparency=255)
            
            # Draw policy visualization
            if best_action in self.__maze.COMPASS:
                self.__draw_policy(dir, best_action)
            
            # move the robot
            self.__temp_robot = self.__robot + np.array(self.__maze.COMPASS[dir])
            # if it's in a portal afterward
            if self.maze.is_portal(self.temp_robot):
                self.__temp_robot = np.array(
                    self.maze.get_portal(tuple(self.temp_robot)).teleport(tuple(self.temp_robot)))

    def reset_robot(self):

        self.__draw_robot(transparency=0)
        self.__robot = np.zeros(2, dtype=int)
        self.__temp_robot = np.zeros(2, dtype=int)
        self.__draw_robot(transparency=255)

    def __controller_update(self):
        if not self.__game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.__game_over = True
                    self.quit_game()

    def __draw_maze(self):
        """绘制迷宫"""
        line_colour = (255, 255, 255, 255)

        # 计算单元格大小
        cell_width = self.maze_area.width / self.maze.MAZE_W
        cell_height = self.maze_area.height / self.maze.MAZE_H

        # 绘制水平线
        for y in range(self.maze.MAZE_H + 1):
            pygame.draw.line(self.maze_layer, line_colour,
                           (self.maze_area.left, self.maze_area.top + y * cell_height),
                           (self.maze_area.right, self.maze_area.top + y * cell_height), width=1)

        # 绘制垂直线
        for x in range(self.maze.MAZE_W + 1):
            pygame.draw.line(self.maze_layer, line_colour,
                           (self.maze_area.left + x * cell_width, self.maze_area.top),
                           (self.maze_area.left + x * cell_width, self.maze_area.bottom), width=1)

        # 打破墙壁
        for x in range(len(self.maze.maze_cells)):
            for y in range(len(self.maze.maze_cells[x])):
                walls_status = self.maze.get_walls_status(self.maze.maze_cells[x, y])
                dirs = ""
                for dir, open in walls_status.items():
                    if open:
                        dirs += dir
                self.__cover_walls(x, y, dirs)

    def __cover_walls(self, x, y, dirs, colour=(0, 128, 0, 255)):
        """覆盖墙壁"""
        cell_width = self.maze_area.width / self.maze.MAZE_W
        cell_height = self.maze_area.height / self.maze.MAZE_H

        dx = self.maze_area.left + x * cell_width
        dy = self.maze_area.top + y * cell_height

        if not isinstance(dirs, str):
            raise TypeError("dirs must be a str.")

        for dir in dirs:
            if dir == "S":
                line_head = (dx + 1, dy + cell_height)
                line_tail = (dx + cell_width - 1, dy + cell_height)
            elif dir == "N":
                line_head = (dx + 1, dy)
                line_tail = (dx + cell_width - 1, dy)
            elif dir == "W":
                line_head = (dx, dy + 1)
                line_tail = (dx, dy + cell_height - 1)
            elif dir == "E":
                line_head = (dx + cell_width, dy + 1)
                line_tail = (dx + cell_width, dy + cell_height - 1)
            else:
                raise ValueError("The only valid directions are (N, S, E, W).")

            pygame.draw.line(self.maze_layer, colour, line_head, line_tail, width=1)

    def __draw_robot(self, colour=(0, 0, 150), transparency=255):
        """绘制机器人"""
        cell_width = self.maze_area.width / self.maze.MAZE_W
        cell_height = self.maze_area.height / self.maze.MAZE_H

        x = self.maze_area.left + int(self.__robot[0] * cell_width + cell_width * 0.5 + 0.5)
        y = self.maze_area.top + int(self.__robot[1] * cell_height + cell_height * 0.5 + 0.5)
        r = int(min(cell_width, cell_height) / 5 + 0.5)

        pygame.draw.circle(self.maze_layer, colour + (transparency,), (x, y), r)

    def __draw_entrance(self, colour=(0, 0, 150), transparency=235):
        """绘制入口"""
        self.__colour_cell(self.entrance, colour=colour, transparency=transparency)

    def __draw_goal(self, colour=(150, 0, 0), transparency=235):
        """绘制目标"""
        self.__colour_cell(self.goal, colour=colour, transparency=transparency)

    def __draw_portals(self, transparency=160):

        colour_range = np.linspace(0, 255, len(self.maze.portals), dtype=int)
        colour_i = 0
        for portal in self.maze.portals:
            colour = ((100 - colour_range[colour_i]) % 255, colour_range[colour_i], 0)
            colour_i += 1
            for location in portal.locations:
                self.__colour_cell(location, colour=colour, transparency=transparency)

    ####
    def __draw_policy(self, dir, best_action):
        if self.__last_best_pos is not None:
            self.__clear_cell(self.__last_best_pos)
        if self.__last_chosen_pos is not None:
            self.__clear_cell(self.__last_chosen_pos)
            
        # Current robot position
        cur_pos = self.__robot
        
        # Draw best action position
        if best_action in self.__maze.COMPASS:
            offset2 = np.array(self.__maze.COMPASS[best_action])
            next_pos2 = cur_pos + offset2
            self.__last_best_pos = tuple(next_pos2)
            self.__draw_rect(tuple(next_pos2), colour=(255, 0, 0), label="BEST")
        
        # Draw chosen action position
        if dir in self.__maze.COMPASS:
            offset1 = np.array(self.__maze.COMPASS[dir])
            next_pos1 = cur_pos + offset1
            self.__last_chosen_pos = tuple(next_pos1)
            self.__draw_rect(tuple(next_pos1), colour=(0, 0, 255), label="NEXT")

    ####
    def __clear_cell(self, pos):
        """Clear a cell by restoring its background color.
        
        Args:
            pos: Tuple of (x, y) coordinates
        """
        # Draw background color rectangle
        self.__draw_rect(tuple(pos), colour=(0, 128, 0), label=None)

    def __draw_rect(self, cell, colour, label=None):
        """Draw a colored rectangle with optional label in the specified cell.
        
        Args:
            cell: Tuple of (x, y) coordinates
            colour: RGB color tuple
            label: Optional text label to draw in the cell
        """
        if not isinstance(cell, tuple):
            cell = tuple(cell)

        # Calculate cell dimensions
        cell_width = self.maze_area.width / self.maze.MAZE_W
        cell_height = self.maze_area.height / self.maze.MAZE_H

        # Calculate cell position
        x = self.maze_area.left + cell[0] * cell_width
        y = self.maze_area.top + cell[1] * cell_height

        # Draw colored rectangle
        pygame.draw.rect(self.maze_layer, colour + (128,), 
                       (x + 1, y + 1, cell_width - 2, cell_height - 2))

        # Draw label if provided
        if label is not None:
            try:
                font = pygame.font.Font(None, int(min(cell_width, cell_height) * 0.4))
                text = font.render(label, True, (255, 255, 255))
                text_rect = text.get_rect(center=(x + cell_width/2, y + cell_height/2))
                self.maze_layer.blit(text, text_rect)
            except pygame.error:
                # Skip text rendering if font initialization fails
                pass

    def __colour_cell(self, cell, colour, transparency):
        """为单元格着色"""
        if not (isinstance(cell, (list, tuple, np.ndarray)) and len(cell) == 2):
            raise TypeError("cell must a be a tuple, list, or numpy array of size 2")

        cell_width = self.maze_area.width / self.maze.MAZE_W
        cell_height = self.maze_area.height / self.maze.MAZE_H

        x = self.maze_area.left + int(cell[0] * cell_width + 0.5 + 1)
        y = self.maze_area.top + int(cell[1] * cell_height + 0.5 + 1)
        w = int(cell_width + 0.5 - 1)
        h = int(cell_height + 0.5 - 1)
        pygame.draw.rect(self.maze_layer, colour + (transparency,), (x, y, w, h))

    @property
    def maze(self):
        return self.__maze

    @property
    def robot(self):
        return self.__robot

    @property
    def temp_robot(self):
        return self.__temp_robot

    @property
    def entrance(self):
        return self.__entrance

    @property
    def goal(self):
        return self.__goal

    @property
    def game_over(self):
        return self.__game_over

    @property
    def SCREEN_SIZE(self):
        return tuple(self.__screen_size)

    @property
    def SCREEN_W(self):
        return int(self.SCREEN_SIZE[0])

    @property
    def SCREEN_H(self):
        return int(self.SCREEN_SIZE[1])

    @property
    def CELL_W(self):
        return float(self.SCREEN_W) / float(self.maze.MAZE_W)

    @property
    def CELL_H(self):
        return float(self.SCREEN_H) / float(self.maze.MAZE_H)

    ########
    def update_metrics_plot(self, epoch, reward_value=None, loss_value=None):
        """更新指标图表"""
        if reward_value is not None:
            self.reward_history.append(reward_value)
        if loss_value is not None:
            self.loss_history.append(loss_value)

        # 清除绘图区域
        pygame.draw.rect(self.screen, (255, 255, 255), self.plot_area)

        # 字体设置
        tick_font = pygame.font.SysFont('Arial', 18)
        label_font = pygame.font.SysFont('Arial', 20, bold=True)

        # 图像区域参数
        margin = 50
        x_start = self.plot_area.left + margin
        y_start = self.plot_area.bottom - margin
        width = self.plot_area.width - 2 * margin
        height = self.plot_area.height - 2 * margin

        if len(self.reward_history) > 1:
            # 绘制奖励曲线
            reward_points = []
            max_reward = max(self.reward_history)
            min_reward = min(self.reward_history)
            reward_range = max_reward - min_reward if max_reward != min_reward else 1

            for i, r in enumerate(self.reward_history):
                x = x_start + int(i * (width / max(1, len(self.reward_history) - 1)))
                y = y_start - int((r - min_reward) * (height / reward_range))
                reward_points.append((x, y))

            if len(reward_points) >= 2:
                pygame.draw.lines(self.screen, (0, 0, 255), False, reward_points, 2)

            # 绘制损失曲线
            if self.loss_history:
                loss_points = []
                max_loss = max(self.loss_history)
                min_loss = min(self.loss_history)
                loss_range = max_loss - min_loss if max_loss != min_loss else 1

                for i, l in enumerate(self.loss_history):
                    x = x_start + int(i * (width / max(1, len(self.loss_history) - 1)))
                    y = y_start - int((l - min_loss) * (height / loss_range))
                    loss_points.append((x, y))

                if len(loss_points) >= 2:
                    pygame.draw.lines(self.screen, (255, 0, 0), False, loss_points, 2)

            # 绘制网格线
            for i in range(6):
                # 水平网格线
                y = y_start - int(i * height / 5)
                pygame.draw.line(self.screen, (200, 200, 200), (x_start, y),
                               (x_start + width, y), 1)

                # 垂直网格线
                x = x_start + int(i * width / 5)
                pygame.draw.line(self.screen, (200, 200, 200), (x, y_start - height),
                               (x, y_start), 1)

            # 绘制坐标轴
            pygame.draw.line(self.screen, (0, 0, 0), (x_start, y_start - height),
                         (x_start, y_start), 2)  # Y轴
            pygame.draw.line(self.screen, (0, 0, 0), (x_start, y_start),
                         (x_start + width, y_start), 2)  # X轴

            # Y轴刻度（左侧reward）
            for i in range(6):
                y = y_start - int(i * height / 5)
                val = min_reward + (i * reward_range / 5)
                text = tick_font.render(f'{val:.2f}', True, (0, 0, 255))
                self.screen.blit(text, (x_start - 45, y - 10))

            # Y轴刻度（右侧loss）
            if self.loss_history:
                for i in range(6):
                    y = y_start - int(i * height / 5)
                    val = min_loss + (i * loss_range / 5)
                    text = tick_font.render(f'{val:.2f}', True, (255, 0, 0))
                    text_width = text.get_width()
                    self.screen.blit(text, (x_start + width + 10, y - 10))

            # X轴刻度
            step = max(1, len(self.reward_history) // 5)
            for i in range(0, len(self.reward_history), step):
                x = x_start + int(i * width / max(1, len(self.reward_history) - 1))
                text = tick_font.render(str(i), True, (0, 0, 0))
                self.screen.blit(text, (x - 10, y_start + 10))

            # 标题和图例
            title = label_font.render('Training Progress', True, (0, 0, 0))
            self.screen.blit(title, (x_start + width // 2 - 60, y_start - height - 30))

            # 图例
            legend_y = y_start - height - 30
            pygame.draw.line(self.screen, (0, 0, 255), (x_start + width - 150, legend_y),
                           (x_start + width - 120, legend_y), 2)
            self.screen.blit(label_font.render('Reward', True, (0, 0, 255)),
                           (x_start + width - 110, legend_y - 10))

            if self.loss_history:
                pygame.draw.line(self.screen, (255, 0, 0), (x_start + width - 150, legend_y + 20),
                               (x_start + width - 120, legend_y + 20), 2)
                self.screen.blit(label_font.render('Loss', True, (255, 0, 0)),
                               (x_start + width - 110, legend_y + 10))

        # 更新显示
        # pygame.display.update(self.plot_area)


class Maze:
    COMPASS = {
        "N": (0, -1),
        "E": (1, 0),
        "S": (0, 1),
        "W": (-1, 0)
    }

    def __init__(self, maze_cells=None, maze_size=(10, 10), has_loops=True, num_portals=0):

        # maze member variables
        self.maze_cells = maze_cells
        self.has_loops = has_loops
        self.__portals_dict = dict()
        self.__portals = []
        self.num_portals = num_portals

        # Use existing one if exists
        if self.maze_cells is not None:
            if isinstance(self.maze_cells, (np.ndarray, np.generic)) and len(self.maze_cells.shape) == 2:
                self.maze_size = tuple(maze_cells.shape)
            else:
                raise ValueError("maze_cells must be a 2D NumPy array.")
        # Otherwise, generate a random one
        else:
            # maze's configuration parameters
            if not (isinstance(maze_size, (list, tuple)) and len(maze_size) == 2):
                raise ValueError("maze_size must be a tuple: (width, height).")
            self.maze_size = maze_size

            self._generate_maze()

    def save_maze(self, file_path):

        if not isinstance(file_path, str):
            raise TypeError("Invalid file_path. It must be a str.")

        if not os.path.exists(os.path.dirname(file_path)):
            raise ValueError("Cannot find the directory for %s." % file_path)

        else:
            np.save(file_path, self.maze_cells, allow_pickle=False, fix_imports=True)

    @classmethod
    def load_maze(cls, file_path):

        if not isinstance(file_path, str):
            raise TypeError("Invalid file_path. It must be a str.")

        if not os.path.exists(file_path):
            raise ValueError("Cannot find %s." % file_path)

        else:
            return np.load(file_path, allow_pickle=False, fix_imports=True)

    def _generate_maze(self):

        # list of all cell locations
        self.maze_cells = np.zeros(self.maze_size, dtype=int)

        # Initializing constants and variables needed for maze generation
        current_cell = (random.randint(0, self.MAZE_W - 1), random.randint(0, self.MAZE_H - 1))
        num_cells_visited = 1
        cell_stack = [current_cell]

        # Continue until all cells are visited
        while cell_stack:

            # restart from a cell from the cell stack
            current_cell = cell_stack.pop()
            x0, y0 = current_cell

            # find neighbours of the current cells that actually exist
            neighbours = dict()
            for dir_key, dir_val in self.COMPASS.items():
                x1 = x0 + dir_val[0]
                y1 = y0 + dir_val[1]
                # if cell is within bounds
                if 0 <= x1 < self.MAZE_W and 0 <= y1 < self.MAZE_H:
                    # if all four walls still exist
                    if self.all_walls_intact(self.maze_cells[x1, y1]):
                        #if self.num_walls_broken(self.maze_cells[x1, y1]) <= 1:
                        neighbours[dir_key] = (x1, y1)

            # if there is a neighbour
            if neighbours:
                # select a random neighbour
                dir = random.choice(tuple(neighbours.keys()))
                x1, y1 = neighbours[dir]

                # knock down the wall between the current cell and the selected neighbour
                self.maze_cells[x1, y1] = self.__break_walls(self.maze_cells[x1, y1], self.__get_opposite_wall(dir))

                # push the current cell location to the stack
                cell_stack.append(current_cell)

                # make the this neighbour cell the current cell
                cell_stack.append((x1, y1))

                # increment the visited cell count
                num_cells_visited += 1

        if self.has_loops:
            self.__break_random_walls(0.2)

        if self.num_portals > 0:
            self.__set_random_portals(num_portal_sets=self.num_portals, set_size=2)

    def __break_random_walls(self, percent):
        # find some random cells to break
        num_cells = int(round(self.MAZE_H * self.MAZE_W * percent))
        cell_ids = random.sample(range(self.MAZE_W * self.MAZE_H), num_cells)

        # for each of those walls
        for cell_id in cell_ids:
            x = cell_id % self.MAZE_H
            y = int(cell_id / self.MAZE_H)

            # randomize the compass order
            dirs = random.sample(list(self.COMPASS.keys()), len(self.COMPASS))
            for dir in dirs:
                # break the wall if it's not already open
                if self.is_breakable((x, y), dir):
                    self.maze_cells[x, y] = self.__break_walls(self.maze_cells[x, y], dir)
                    break

    def __set_random_portals(self, num_portal_sets, set_size=2):
        # find some random cells to break
        num_portal_sets = int(num_portal_sets)
        set_size = int(set_size)

        # limit the maximum number of portal sets to the number of cells available.
        max_portal_sets = int(self.MAZE_W * self.MAZE_H / set_size)
        num_portal_sets = min(max_portal_sets, num_portal_sets)

        # the first and last cells are reserved
        cell_ids = random.sample(range(1, self.MAZE_W * self.MAZE_H - 1), num_portal_sets * set_size)

        for i in range(num_portal_sets):
            # sample the set_size number of sell
            portal_cell_ids = random.sample(cell_ids, set_size)
            portal_locations = []
            for portal_cell_id in portal_cell_ids:
                # remove the cell from the set of potential cell_ids
                cell_ids.pop(cell_ids.index(portal_cell_id))
                # convert portal ids to location
                x = portal_cell_id % self.MAZE_H
                y = int(portal_cell_id / self.MAZE_H)
                portal_locations.append((x, y))
            # append the new portal to the maze
            portal = Portal(*portal_locations)
            self.__portals.append(portal)

            # create a dictionary of portals
            for portal_location in portal_locations:
                self.__portals_dict[portal_location] = portal

    def is_open(self, cell_id, dir):
        # check if it would be out-of-bound
        x1 = cell_id[0] + self.COMPASS[dir][0]
        y1 = cell_id[1] + self.COMPASS[dir][1]

        # if cell is still within bounds after the move
        if self.is_within_bound(x1, y1):
            # check if the wall is opened
            this_wall = bool(self.get_walls_status(self.maze_cells[cell_id[0], cell_id[1]])[dir])
            other_wall = bool(self.get_walls_status(self.maze_cells[x1, y1])[self.__get_opposite_wall(dir)])
            return this_wall or other_wall
        return False

    def is_breakable(self, cell_id, dir):
        # check if it would be out-of-bound
        x1 = cell_id[0] + self.COMPASS[dir][0]
        y1 = cell_id[1] + self.COMPASS[dir][1]

        return not self.is_open(cell_id, dir) and self.is_within_bound(x1, y1)

    def is_within_bound(self, x, y):
        # true if cell is still within bounds after the move
        return 0 <= x < self.MAZE_W and 0 <= y < self.MAZE_H

    def is_portal(self, cell):
        return tuple(cell) in self.__portals_dict

    @property
    def portals(self):
        return tuple(self.__portals)

    def get_portal(self, cell):
        if cell in self.__portals_dict:
            return self.__portals_dict[cell]
        return None

    @property
    def MAZE_W(self):
        return int(self.maze_size[0])

    @property
    def MAZE_H(self):
        return int(self.maze_size[1])

    @classmethod
    def get_walls_status(cls, cell):
        walls = {
            "N": (cell & 0x1) >> 0,
            "E": (cell & 0x2) >> 1,
            "S": (cell & 0x4) >> 2,
            "W": (cell & 0x8) >> 3,
        }
        return walls

    @classmethod
    def all_walls_intact(cls, cell):
        return cell & 0xF == 0

    @classmethod
    def num_walls_broken(cls, cell):
        walls = cls.get_walls_status(cell)
        num_broken = 0
        for wall_broken in walls.values():
            num_broken += wall_broken
        return num_broken

    @classmethod
    def __break_walls(cls, cell, dirs):
        if "N" in dirs:
            cell |= 0x1
        if "E" in dirs:
            cell |= 0x2
        if "S" in dirs:
            cell |= 0x4
        if "W" in dirs:
            cell |= 0x8
        return cell

    @classmethod
    def __get_opposite_wall(cls, dirs):

        if not isinstance(dirs, str):
            raise TypeError("dirs must be a str.")

        opposite_dirs = ""

        for dir in dirs:
            if dir == "N":
                opposite_dir = "S"
            elif dir == "S":
                opposite_dir = "N"
            elif dir == "E":
                opposite_dir = "W"
            elif dir == "W":
                opposite_dir = "E"
            else:
                raise ValueError("The only valid directions are (N, S, E, W).")

            opposite_dirs += opposite_dir

        return opposite_dirs


class Portal:

    def __init__(self, *locations):

        self.__locations = []
        for location in locations:
            if isinstance(location, (tuple, list)):
                self.__locations.append(tuple(location))
            else:
                raise ValueError("location must be a list or a tuple.")

    def teleport(self, cell):
        if cell in self.locations:
            return self.locations[(self.locations.index(cell) + 1) % len(self.locations)]
        return cell

    def get_index(self, cell):
        return self.locations.index(cell)

    @property
    def locations(self):
        return self.__locations


if __name__ == "__main__":
    maze = MazeView2D(screen_size=(500, 500), maze_size=(10, 10))
    maze.update()
    input("Enter any key to quit.")
