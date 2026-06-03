from fpdf import FPDF
import matplotlib.pyplot as plt

def makepdf(vis_img, IR_img, fused_img, hist_vis, hist_ir, hist_fused, data, name_of_file):
    pdf = FPDF('P', 'mm', 'A4')
    pdf.add_page()
    pdf.set_font('helvetica', '', 16)
    pdf.cell(0, 15, "IMAGE FUSION REPORT")
    pdf.ln()
    pdf.set_font('helvetica', '', 11)
    pdf.cell(80, 8, text="Visible Image")
    pdf.cell(80, 8, text="IR Image")
    pdf.ln()
    cell_width = 70
    cell_height = 50

    x_pos = pdf.get_x()
    y_pos = pdf.get_y()

    pdf.image(vis_img, x=x_pos + 1.5, y=y_pos + 1.5, w=67)
    pdf.cell(cell_width, cell_height, text="", border=True)
    x_pos = x_pos + cell_width + 10 

    pdf.set_x(x_pos)
    
    pdf.image(IR_img, x=x_pos + 1.5, y=y_pos + 1.5, w=67)
    pdf.cell(cell_width, cell_height, text="", border=True)
    pdf.ln(54)

    pdf.set_font('helvetica', '', 11)
    pdf.cell(100, 6, text="The Fused Image")
    pdf.cell(80, 6, text="image Histogram", ln=True)
    
    x_pos = pdf.get_x()
    y_pos = pdf.get_y()
    pdf.image(fused_img, x=x_pos + 1.5, y=y_pos + 1.5, w=87, h=57)
    pdf.cell(90, 60, text="", border=True)

    x_pos = pdf.get_x()
    y_pos = pdf.get_y()
    
    chart_maker(hist_vis, hist_ir, hist_fused)
    
    pdf.image("graph.png", x=x_pos + 1.5, y=y_pos + 1.5, w=87, h=57)
    pdf.cell(90, 60, text="", border=True)
    pdf.ln(63)
    
    pdf.set_font('helvetica', '', 11)
    pdf.cell(0, 8, text="Quality Metric Table")

    for row in data:
        for cell_text in row:
            pdf.cell(45, 8, text=str(cell_text), border=True)
        pdf.ln()
    
    pdf.output(name_of_file)


def chart_maker(hist_vis, hist_ir, hist_fused):
    plt.figure(figsize=(10, 6))

    plt.plot(hist_ir, label="Infrared (IR)", color="red", linewidth=2)
    plt.plot(hist_vis, label="Visible", color="blue", linewidth=2)
    plt.plot(hist_fused, label="Fused", color="purple", linewidth=2.5)

    plt.title("Intensity vs. Pixel Count", fontsize=14, fontweight="bold")
    plt.xlabel("Pixel Intensity (0 - 255)", fontsize=12)
    plt.ylabel("Pixel Count", fontsize=12)
    plt.xlim([0, 255])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=11)

    plt.savefig("graph.png", dpi=300, bbox_inches="tight")
    plt.close() 
    return 


    
