def update_reward_plot(self, epoch, total_reward):
    self.reward_history.append(total_reward)
    # 绘制区域背景
    pygame.draw.rect(self.screen, (255, 255, 255), self.plot_area)

    if len(self.reward_history) > 1:
        max_reward = max(self.reward_history)
        min_reward = min(self.reward_history)
        reward_range = max_reward - min_reward if max_reward != min_reward else 1

        # 折线图起始点和缩放
        x_start = self.plot_area.left + 10
        y_start = self.plot_area.bottom - 10
        width = self.plot_area.width - 20
        height = self.plot_area.height - 20

        points = []
        for i, r in enumerate(self.reward_history):
            x = x_start + int(i * (width / max(1, len(self.reward_history) - 1)))
            y = y_start - int((r - min_reward) * (height / reward_range))
            points.append((x, y))

        # 绘制折线
        if len(points) >= 2:
            pygame.draw.lines(self.screen, (0, 0, 255), False, points, 2)

    # 绘制坐标轴
    pygame.draw.line(self.screen, (0, 0, 0), (self.plot_area.left + 10, self.plot_area.top + 10),
                     (self.plot_area.left + 10, self.plot_area.bottom - 10), 2)
    pygame.draw.line(self.screen, (0, 0, 0), (self.plot_area.left + 10, self.plot_area.bottom - 10),
                     (self.plot_area.right - 10, self.plot_area.bottom - 10), 2)

    # 更新显示
    pygame.display.update(self.plot_area)