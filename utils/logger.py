from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
import os
from datetime import datetime

# 注册中文字体
pdfmetrics.registerFont(TTFont('SimHei', 'utils/simhei.ttf'))  # 你可以改成其他 ttf 文件


def generate_experiment_report_pdf(trainer, notes: str = ""):
    report_path = os.path.join(trainer.save_path, "experiment_report.pdf")
    c = canvas.Canvas(report_path, pagesize=A4)
    width, height = A4
    y = height - 2 * cm

    def draw_line(text, offset=1, font_size=12, bold=False):
        nonlocal y
        if y < 2 * cm:
            c.showPage()
            y = height - 2 * cm
        font_name = "SimHei"  # 使用中文字体
        c.setFont(font_name, font_size)
        c.drawString(2 * cm, y, text)
        y -= offset * cm

    # 内容开始
    draw_line("强化学习实验报告", font_size=16)
    draw_line(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", offset=1)

    draw_line("实验设置", font_size=14)
    draw_line(f"- 环境名：{trainer.env.__class__.__name__}")
    draw_line(f"- 智能体：{trainer.agent.__class__.__name__}")
    draw_line(f"- 训练轮数（episodes）：{trainer.episodes}")
    draw_line(f"- 每轮最大步数（max_steps）：{trainer.max_steps}")

    draw_line("实验结果", font_size=14)
    avg_reward = sum(trainer.reward_history) / len(trainer.reward_history)
    best_reward = max(trainer.reward_history)
    best_episode = trainer.reward_history.index(best_reward) + 1
    avg_steps = sum(trainer.step_used) / len(trainer.step_used)

    draw_line(f"- 平均总回报：{avg_reward:.2f}")
    draw_line(f"- 最佳回报：{best_reward:.2f}（第 {best_episode} 轮）")
    draw_line(f"- 平均步数：{avg_steps:.2f}")
    draw_line(f"- 回报图保存在：reward_curve.png")
    draw_line(f"- Q 表策略图保存在：q_table_policy.png")

    draw_line("参数调整与实验观察", font_size=14)
    if notes:
        for line in notes.split("\n"):
            draw_line(line)
    else:
        draw_line("本次实验使用默认参数，可尝试调整学习率或 epsilon 策略改善效果。")

    draw_line("--- 实验报告自动生成完毕 ---", offset=1.5)
    c.save()
    print(f"✅ 实验报告已保存为 PDF：{report_path}")




