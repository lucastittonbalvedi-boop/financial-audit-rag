"""Script to generate a realistic financial PDF statement with tables for testing and demo."""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def create_sample_financial_pdf(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=8
    )
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontSize=11,
        textColor=colors.HexColor('#1F2937'),
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        spaceAfter=6
    )

    story = []

    # Page 1: Relatório da Administração
    story.append(Paragraph("PETRÓLEO BRASILEIRO S.A. - PETROBRAS", title_style))
    story.append(Paragraph("DEMONSTRAÇÕES FINANCEIRAS PADRONIZADAS (DFP) - EXERCÍCIO DE 2023", h2_style))
    story.append(Paragraph("<b>Relatório da Administração e Mensagem da Diretoria</b>", h2_style))
    story.append(Paragraph(
        "No exercício de 2023, a Petrobras apresentou sólido desempenho operacional e financeiro. "
        "A Receita Líquida de Vendas atingiu R$ 511.994 milhões em 2023, comparada a R$ 641.256 milhões em 2022. "
        "O EBITDA Ajustado atingiu R$ 262.238 milhões em 2023, com margem EBITDA de 51,2%. "
        "O Lucro Líquido totalizou R$ 124.606 milhões no exercício de 2023. "
        "A Dívida Líquida encerrou o ano em R$ 220.150 milhões, com alavancagem de 0,84x.",
        body_style
    ))
    story.append(PageBreak())

    # Page 2: DRE com Tabela
    story.append(Paragraph("Demonstração do Resultado do Exercício (DRE) Consolidada", h2_style))
    story.append(Paragraph("Exercícios findos em 31 de dezembro de 2023 e 2022 (Em milhões de Reais - R$)", body_style))
    
    dre_data = [
        ["Linha da DRE", "2023 (R$ mi)", "2022 (R$ mi)", "Variação %"],
        ["Receita Líquida de Vendas", "511.994", "641.256", "-20,2%"],
        ["Custo dos Produtos Vendidos (CPV)", "(268.420)", "(308.150)", "-12,9%"],
        ["Lucro Bruto", "243.574", "333.106", "-26,9%"],
        ["Despesas com Vendas e Distribuição", "(22.140)", "(20.980)", "+5,5%"],
        ["Despesas Gerais e Administrativas", "(11.850)", "(10.920)", "+8,5%"],
        ["Lucro Operacional (EBIT)", "196.944", "309.726", "-36,4%"],
        ["Resultado Financeiro Líquido", "(32.180)", "(38.450)", "-16,3%"],
        ["Lucro Antes dos Tributos (LAIR)", "164.764", "271.276", "-39,3%"],
        ["Lucro Líquido do Exercício", "124.606", "188.478", "-33,9%"]
    ]
    t_dre = Table(dre_data, colWidths=[200, 100, 100, 80])
    t_dre.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E5E7EB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#111827')),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#9CA3AF')),
    ]))
    story.append(t_dre)
    story.append(PageBreak())

    # Page 3: Nota 14 - Provisões
    story.append(Paragraph("Nota Explicativa 14 - Provisões para Litígios e Contingências", h2_style))
    story.append(Paragraph(
        "A Companhia reconhece provisões para processos judiciais com probabilidade de perda provável conforme CPC 25. "
        "O total das provisões para litígios reconhecidas em 31/12/2023 totalizou R$ 63.140 milhões.",
        body_style
    ))
    
    prov_data = [
        ["Natureza do Litígio", "31/12/2023 (R$ mi)", "31/12/2022 (R$ mi)", "Variação %"],
        ["Litígios Fiscais e Tributários", "34.820", "31.450", "+10,7%"],
        ["Litígios Trabalhistas", "18.450", "19.210", "-4,0%"],
        ["Litígios Cíveis e Ambientais", "9.870", "8.650", "+14,1%"],
        ["Total das Provisões para Litígios", "63.140", "59.310", "+6,5%"]
    ]
    t_prov = Table(prov_data, colWidths=[200, 100, 100, 80])
    t_prov.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E5E7EB')),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#9CA3AF')),
    ]))
    story.append(t_prov)

    doc.build(story)
    print(f"Sample financial PDF successfully created at: {output_path}")


if __name__ == "__main__":
    out_file = Path(__file__).resolve().parent / "petrobras_dfp_2023_sample.pdf"
    create_sample_financial_pdf(out_file)
