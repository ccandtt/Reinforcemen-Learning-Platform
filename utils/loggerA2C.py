from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
from torch import nn
font_path = os.path.join(os.path.dirname(__file__), "simhei.ttf")
pdfmetrics.registerFont(TTFont("SimHei", font_path))

def generate_experiment_report_pdf(trainer, notes: str = ""):
    report_path = os.path.join(os.getcwd(), "result", "a2c", "experiment_report.pdf")
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

    draw_line("强化学习实验报告", font_size=16)
    draw_line(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", offset=1)

    # 实验基本信息
    draw_line("实验设置", font_size=14)
    draw_line(f"- 环境名：{trainer.env.__class__.__name__}")
    draw_line(f"- 智能体：{trainer.agent.__class__.__name__}")
    draw_line(f"- 训练轮数（episodes）：{trainer.episodes}")
    draw_line(f"- 每轮最大步数（max_steps）：{trainer.max_steps}")

    # 网络结构信息（假设ActorCritic里有fc，actor，critic）
    agent = trainer.agent
    try:
        net = agent.policy
        hidden_layer_info = []
        for layer in net.fc:
            if isinstance(layer, nn.Linear):
                hidden_layer_info.append(f"Linear({layer.in_features} → {layer.out_features})")
            elif isinstance(layer, nn.ReLU):
                hidden_layer_info.append("ReLU()")
        draw_line("网络结构", font_size=14)
        draw_line("- ActorCritic 网络隐藏层：")
        for l in hidden_layer_info:
            draw_line(f"  {l}")
        draw_line(f"- Actor 输出层: Linear({net.actor.in_features} → {net.actor.out_features})")
        draw_line(f"- Critic 输出层: Linear({net.critic.in_features} → {net.critic.out_features})")
    except Exception:
        draw_line("网络结构信息获取失败", font_size=12)

    # A2C超参数
    draw_line("A2C 算法超参数", font_size=14)
    for key in ["gamma", "lr", "n_step", "use_td_target", "gae_lambda"]:
        value = getattr(agent, key, None)
        if value is not None:
            draw_line(f"- {key}：{value}")

    # 训练结果统计
    draw_line("训练结果", font_size=14)
    if trainer.reward_history:
        avg_reward = sum(trainer.reward_history) / len(trainer.reward_history)
        best_reward = max(trainer.reward_history)
        best_episode = trainer.reward_history.index(best_reward) + 1
    else:
        avg_reward = best_reward = best_episode = 0

    if trainer.step_used:
        avg_steps = sum(trainer.step_used) / len(trainer.step_used)
    else:
        avg_steps = 0

    draw_line(f"- 平均总回报：{avg_reward:.2f}")
    draw_line(f"- 最佳回报：{best_reward:.2f}（第 {best_episode} 轮）")
    draw_line(f"- 平均步数：{avg_steps:.2f}")

    # 备注或实验观察
    draw_line("参数调整与实验观察", font_size=14)
    if notes:
        for line in notes.split("\n"):
            draw_line(line)
    else:
        draw_line("本次实验使用默认参数，可根据需要调整学习率、n_step等。")

    draw_line("--- 实验报告自动生成完毕 ---", offset=1.5)
    c.save()
    print(f"✅ 实验报告已保存为 PDF：{report_path}")
