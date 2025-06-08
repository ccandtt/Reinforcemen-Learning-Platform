import os
from datetime import datetime
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

# 注册中文字体
pdfmetrics.registerFont(TTFont('SimHei', 'utils/simhei.ttf'))  # 请确保字体文件路径正确

def generate_experiment_report_pdf(trainer, notes: str = ""):
    # 结果保存路径
    report_path = os.path.join(os.getcwd(), "result", "ddpg", "experiment_report.pdf")
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
    draw_line("DDPG 强化学习实验报告", font_size=16)
    draw_line(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", offset=1)

    # 实验设置
    draw_line("实验设置", font_size=14)
    draw_line(f"- 环境名：{trainer.env.spec.id if hasattr(trainer.env, 'spec') else type(trainer.env).__name__}")
    draw_line(f"- 算法：{trainer.agent.__class__.__name__}")
    draw_line(f"- 训练轮数（episodes）：{trainer.episodes}")
    draw_line(f"- 每轮最大步数（max_steps）：{trainer.max_steps}")

    # 超参数
    draw_line("算法超参数", font_size=14)
    agent = trainer.agent

    # 学习率（假设用的是Adam）
    lr_actor = agent.actor_optimizer.param_groups[0]['lr']
    lr_critic = agent.critic_optimizer.param_groups[0]['lr']
    draw_line(f"- Actor学习率：{lr_actor}")
    draw_line(f"- Critic学习率：{lr_critic}")
    draw_line(f"- 折扣因子（gamma）：{agent.gamma}")
    draw_line(f"- 软更新系数（tau）：{agent.tau}")
    draw_line(f"- 最大动作范围（max_action）：{agent.max_action}")

    # 经验回放设置
    draw_line(f"- 是否使用多步TD：{'是' if agent.replay_buffer.n_step > 1 else '否'}")
    draw_line(f"- 多步数值：{agent.replay_buffer.n_step}")
    draw_line(f"- 是否使用优先级经验重放：{'是' if agent.use_per else '否'}")

    # 网络结构简单描述
    draw_line(f"- Actor网络结构：{agent.actor.net}")
    draw_line(f"- Critic网络结构：{agent.critic.net}")

    # 实验结果
    draw_line("实验结果", font_size=14)
    avg_reward = sum(trainer.reward_history) / len(trainer.reward_history) if trainer.reward_history else 0
    best_reward = max(trainer.reward_history) if trainer.reward_history else 0
    best_episode = trainer.reward_history.index(best_reward) + 1 if trainer.reward_history else 0
    avg_steps = sum(trainer.step_used) / len(trainer.step_used) if trainer.step_used else 0

    draw_line(f"- 平均总回报：{avg_reward:.2f}")
    draw_line(f"- 最佳回报：{best_reward:.2f}（第 {best_episode} 轮）")
    draw_line(f"- 平均步数：{avg_steps:.2f}")

    # 额外信息（比如图像路径）
    draw_line(f"- 奖励曲线图保存在：reward_curve.png")
    draw_line(f"- 策略图保存在：policy_map.png")

    # 备注
    draw_line("参数调整与实验观察", font_size=14)
    if notes:
        for line in notes.split("\n"):
            draw_line(line)
    else:
        draw_line("本次实验使用默认参数，可尝试调整学习率、tau等以优化训练效果。")

    draw_line("--- 实验报告自动生成完毕 ---", offset=1.5)
    c.save()
    print(f"✅ DDPG实验报告已保存为 PDF：{report_path}")
