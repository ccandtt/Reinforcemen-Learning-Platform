import os
from datetime import datetime
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

# 注册中文字体
pdfmetrics.registerFont(TTFont('SimHei', 'utils/simhei.ttf'))

def generate_experiment_report_pdf(trainer, notes: str = ""):
    report_path = os.path.join(os.getcwd(), "result", "ppo", "experiment_report.pdf")
    if not os.path.exists(os.path.dirname(report_path)):
        os.makedirs(os.path.dirname(report_path))

    c = canvas.Canvas(report_path, pagesize=A4)
    width, height = A4
    y = height - 2 * cm

    def draw_line(text, offset=1, font_size=12):
        nonlocal y
        if y < 2 * cm:
            c.showPage()
            y = height - 2 * cm
        c.setFont("SimHei", font_size)
        c.drawString(2 * cm, y, text)
        y -= offset * cm

    # 标题
    draw_line("强化学习实验报告 - PPO", font_size=16)
    draw_line(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", offset=1)

    # 实验设置
    draw_line("实验设置", font_size=14)
    draw_line(f"- 环境名：{trainer.env.__class__.__name__}")
    draw_line(f"- 智能体：{trainer.agent.__class__.__name__}")
    draw_line(f"- 训练轮数（episodes）：{trainer.episodes}")
    draw_line(f"- 每轮最大步数（max_steps）：{trainer.max_steps}")

    # PPO算法超参数
    draw_line("PPO算法超参数", font_size=14)
    agent = trainer.agent

    lr = agent.optimizer.param_groups[0]['lr'] if hasattr(agent, 'optimizer') else "未知"
    draw_line(f"- 学习率（lr）：{lr}")
    draw_line(f"- 折扣因子（gamma）：{getattr(agent, 'gamma', '未知')}")
    draw_line(f"- clip参数（clip_epsilon）：{getattr(agent, 'clip_epsilon', '未知')}")
    draw_line(f"- 熵系数（entropy_coef）：{getattr(agent, 'entropy_coef', '未知')}")
    draw_line(f"- 是否使用 n-step TD（use_nstep_td）：{getattr(agent, 'use_nstep_td', False)}")
    if getattr(agent, 'use_nstep_td', False):
        draw_line(f"- n-step 的步数（n_step）：{getattr(agent, 'n_step', '未知')}")

    # 网络结构隐藏层维度
    hidden_dims = getattr(agent.policy, 'hidden_dims', '未知')
    draw_line(f"- 网络结构（隐藏层维度）：{hidden_dims}")

    # 训练结果
    draw_line("实验结果", font_size=14)
    if trainer.reward_history and len(trainer.reward_history) > 0:
        avg_reward = sum(trainer.reward_history) / len(trainer.reward_history)
        best_reward = max(trainer.reward_history)
        best_episode = trainer.reward_history.index(best_reward) + 1
        avg_steps = sum(trainer.step_used) / len(trainer.step_used) if trainer.step_used else 0

        draw_line(f"- 平均总回报：{avg_reward:.2f}")
        draw_line(f"- 最佳回报：{best_reward:.2f}（第 {best_episode} 轮）")
        draw_line(f"- 平均步数：{avg_steps:.2f}")
    else:
        draw_line("- 无奖励数据")

    # 图像文件路径提示（如果你有生成对应图像，可修改文件名）
    draw_line(f"- 奖励曲线图保存在：reward_curve.png")
    draw_line(f"- 策略图保存在：policy_map.png")

    # 备注
    draw_line("参数调整与实验观察", font_size=14)
    if notes:
        for line in notes.split("\n"):
            draw_line(line)
    else:
        draw_line("本次实验使用默认参数，尝试调整学习率、熵系数或 n-step 参数以优化效果。")

    draw_line("--- 实验报告自动生成完毕 ---", offset=1.5)
    c.save()
    print(f"✅ 实验报告已保存为 PDF：{report_path}")
