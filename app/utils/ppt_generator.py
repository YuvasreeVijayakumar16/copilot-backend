import uuid
from pptx import Presentation
from pptx.util import Inches
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from datetime import datetime
import os
import pandas as pd
from docx import Document
from openai import OpenAI
import re
import matplotlib.pyplot as plt
from pptx.util import Pt

from pptx.enum.chart import XL_CHART_TYPE
from pptx.chart.data import CategoryChartData
 
import time
from pptx.enum.text import PP_ALIGN
 
import numpy as np
 

# === ✅ Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === ✅ Generate GPT Insights and Recommendations
def generate_insights(df: pd.DataFrame):
    prompt = f"Analyze this data and provide 3–5 business insights and 2–3 strong recommendations:\n\n{df.head(10).to_string(index=False)}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a data analyst generating insights for supply chain decision makers."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    output = response.choices[0].message.content.strip()

    # === ✅ Cleanup: remove markdown bold/italic and numbered bullets
    output = re.sub(r"\*\*(.*?)\*\*", r"\1", output)  # Remove bold
    output = re.sub(r"\*(.*?)\*", r"\1", output)      # Remove italic
    output = re.sub(r"^\s*\d+\.\s*", "", output, flags=re.MULTILINE)  # Remove numbered list

    parts = re.split(r"\n{2,}", output)
    insights = ""
    recs = ""

    for part in parts:
        if "recommend" in part.lower():
            recs = part.strip()
        else:
            insights = part.strip()

    return insights, recs

# === ✅ Create simple bar chart image
def create_bar_chart(df: pd.DataFrame, column: str = None, output_path: str = "generated_files/bar_chart.png") -> str:
    if not column:
        # Auto-pick first categorical column
        for col in df.columns:
            if df[col].dtype == "object" or df[col].nunique() < 20:
                column = col
                break
    if not column:
        return None  # No suitable column found

    chart_data = df[column].value_counts().head(5)
    plt.figure(figsize=(6, 4))
    chart_data.plot(kind="bar", color="skyblue")
    plt.title(f"Top 5 {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()

    os.makedirs("generated_files", exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path

def generate_unique_blob_name(prefix="PPT", actions_list=None, extension="pptx"):
    """
    Generate a unique filename using multiple actions from the message.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = uuid.uuid4().hex
    if actions_list:
        # Keep only alphanumeric and combine first two descriptive actions
        snippet = "_".join([''.join(e for e in act if e.isalnum())[:20] for act in actions_list[:2]])
    else:
        snippet = "Output"
    return f"{prefix}_{snippet}_{timestamp}_{unique_id}.{extension}"
 
def extract_actions(message):
    """Extract multiple actionable parts from the message."""
    parts = [p.strip() for p in re.split(r"\band\b|,|\s{2,}", message) if p.strip()]
    return parts
 
def select_descriptive_action(actions):
    """
    Pick the most descriptive action (longest meaningful string) for title.
    """
    if not actions:
        return ""
    # Prefer parts longer than 10 characters
    sorted_actions = sorted(actions, key=lambda x: len(x), reverse=True)
    for act in sorted_actions:
        if len(act) > 10:
            return act
    return actions[0]
 
# ------------------ Main PPT Generator ------------------
 
def generate_ppt(message: str, df: pd.DataFrame, include_charts: bool = False):
    """
    Generate a fully dynamic PPT based on the message and dataframe.
    Returns only the file path string for easier upload handling.
    """
    actions = extract_actions(message)
    descriptive_action = select_descriptive_action(actions)  # for title
    additional_actions = [a for a in actions if a != descriptive_action]  # remaining for insights

    prs = Presentation()

    # ---------- Title Slide ----------
    slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = "🚀 Supply Sense AI"
    slide.placeholders[1].text = descriptive_action

    # ---------- Insights & Recommendations ----------
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(exclude='number').columns.tolist()

    insights, recs = [], []

    for col in numeric_cols[:5]:  # Limit to first 5 numeric cols to prevent overflow
        insights.append(f"📊 '{col}': min={df[col].min()}, max={df[col].max()}, avg={df[col].mean():.2f}")
        recs.append(f"✅ Optimize '{col}' based on trends.")

    for col in categorical_cols[:3]:  # Limit to first 3 categorical cols
        top_items = ', '.join(df[col].value_counts().head(3).index.astype(str))
        insights.append(f"🌟 Top '{col}': {top_items}")
        recs.append(f"✅ Focus on top '{col}' items: {top_items}.")

    # Include remaining actions as bullet points, limit to 5
    if additional_actions:
        insights.extend([f"🔹 {act}" for act in additional_actions[:5]])

    slide_layout = prs.slide_layouts[1] if len(prs.slide_layouts) > 1 else prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = "📊 Insights & Recommendations"
    tf = slide.placeholders[1].text_frame
    tf.clear()
    tf.word_wrap = True  # Enable word wrap to prevent overflow
    p = tf.add_paragraph()
    p.text = '\n'.join(insights)
    p.font.size = Pt(12)  # Reduced font size
    p = tf.add_paragraph()
    p.text = '\n'.join(recs)
    p.font.size = Pt(12)  # Reduced font size

    # ---------- Chart Slides ----------
    if include_charts and numeric_cols:
        for col in numeric_cols:
            plt.figure(figsize=(6,4))
            plt.bar(df.index.astype(str), df[col], color="#4CAF50")
            plt.title(f"{col} Analysis", fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            chart_path = f"chart_{uuid.uuid4().hex[:8]}.png"
            plt.savefig(chart_path)
            plt.close()
            slide_layout = prs.slide_layouts[5] if len(prs.slide_layouts) > 5 else prs.slide_layouts[0]
            slide = prs.slides.add_slide(slide_layout)
            slide.shapes.add_picture(chart_path, Inches(1), Inches(1.2), width=Inches(7))
            title_shape = slide.shapes.title if slide.shapes.title else slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(0.2), Inches(8), Inches(0.5)
            )
            title_shape.text = f"📈 {col} Chart"
            if os.path.exists(chart_path):
                os.remove(chart_path)

    # ---------- Data Preview Slide ----------
    slide_layout = prs.slide_layouts[5] if len(prs.slide_layouts) > 5 else prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = "🗂 Data Preview"
    # Limit columns to 6 to prevent overflow
    display_cols = df.columns[:6]
    rows, cols = min(10, len(df)) + 1, len(display_cols)
    if cols > 0:
        table = slide.shapes.add_table(rows, cols, Inches(0.5), Inches(1.5), Inches(9), Inches(0.8)).table

        for col_idx, col_name in enumerate(display_cols):
            table.cell(0, col_idx).text = str(col_name)
        for row_idx, row in df.head(10).iterrows():
            for col_idx, col in enumerate(display_cols):
                table.cell(row_idx + 1, col_idx).text = str(row[col])
    else:
        # Add text box for no data
        text_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(9), Inches(0.8))
        text_frame = text_box.text_frame
        text_frame.text = "No data available for preview."

    # ---------- Highlights Slide ----------
    slide_layout = prs.slide_layouts[1] if len(prs.slide_layouts) > 1 else prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = "📖 Key Highlights"
    tf = slide.placeholders[1].text_frame
    tf.clear()
    tf.word_wrap = True  # Enable word wrap to prevent overflow
    highlights = []

    for col in numeric_cols[:5]:  # Limit to first 5 numeric cols
        highlights.append(f"📌 Max {col}: {df[col].max()}, Min {col}: {df[col].min()}")
    for col in categorical_cols[:3]:  # Limit to first 3 categorical cols
        top_items = ', '.join(df[col].value_counts().head(3).index.astype(str))
        highlights.append(f"📌 Top '{col}': {top_items}")

    for h in highlights:
        p = tf.add_paragraph()
        p.text = h
        p.font.size = Pt(12)  # Reduced font size

    # ---------- Thank You Slide ----------
    slide_layout = prs.slide_layouts[1] if len(prs.slide_layouts) > 1 else prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = "🙏 Thank You"
    slide.placeholders[1].text = "This presentation was auto-generated dynamically."

    # ---------- Save PPT ----------
    output_dir = "generated_files"
    os.makedirs(output_dir, exist_ok=True)
    # Fix: Use correct generate_unique_blob_name signature
    filename = generate_unique_blob_name(prefix="PPT", actions_list=actions, extension="pptx")
    ppt_path = os.path.join(output_dir, filename)
    prs.save(ppt_path)

    print(f"✅ Presentation generated for message: '{message}'")
    return ppt_path

def generate_excel(df: pd.DataFrame, question: str, include_charts: bool = False) -> str:
    """
    Generate an Excel file from the given DataFrame.
    """
    filename = f"{question[:30].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"
    output_dir = "generated_files"
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, filename)
    df.to_excel(excel_path, index=False)
    return excel_path

def generate_word(df: pd.DataFrame, question: str, include_charts: bool = False) -> str:
    """
    Generate a Word document from the given DataFrame.
    """
    doc = Document()
    doc.add_heading(question, 0)
    
    # Add data table
    table = doc.add_table(rows=1, cols=len(df.columns))
    hdr_cells = table.rows[0].cells
    for i, col in enumerate(df.columns):
        hdr_cells[i].text = str(col)
    
    for _, row in df.head(10).iterrows():
        row_cells = table.add_row().cells
        for i, col in enumerate(df.columns):
            row_cells[i].text = str(row[col])
    
    # Add insights
    insights, recs = generate_insights(df)
    doc.add_heading('Insights', level=1)
    doc.add_paragraph(insights)
    doc.add_heading('Recommendations', level=1)
    doc.add_paragraph(recs)
    
    filename = f"{question[:30].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.docx"
    output_dir = "generated_files"
    os.makedirs(output_dir, exist_ok=True)
    word_path = os.path.join(output_dir, filename)
    doc.save(word_path)
    return word_path

def generate_direct_response(question: str, df: pd.DataFrame) -> str:
    """
    Generate a direct response to the question based on the data.
    """
    prompt = f"Answer the following question based on the data provided:\n\nQuestion: {question}\n\nData:\n{df.head(10).to_string(index=False)}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant providing direct answers based on data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content.strip()
    return answer

def set_background_color(slide, rgb_tuple=(255, 255, 255)):
    slide_background = slide.background
    fill = slide_background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(*rgb_tuple)
 
 
def add_title_slide(prs, title, subtitle, bg_color=(255, 255, 255)):
    """Add a professional title slide"""
    slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)
    set_background_color(slide, bg_color)

    # Set title with adjusted font size for overflow fix
    title_shape = slide.shapes.title
    title_shape.text = title
    title_shape.text_frame.paragraphs[0].font.size = Pt(32)  # Adjusted to prevent overflow
    title_shape.text_frame.paragraphs[0].font.bold = True

    # Set subtitle
    subtitle_shape = slide.placeholders[1]
    subtitle_shape.text = subtitle
    subtitle_shape.text_frame.paragraphs[0].font.size = Pt(20)
    subtitle_shape.text_frame.paragraphs[0].font.color.rgb = RGBColor(100, 100, 100)

    # Add date and footer
    footer_text = f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(6.5), Inches(9), Inches(0.5))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = footer_text
    p.font.size = Pt(10)
    p.font.color.rgb = RGBColor(150, 150, 150)


def add_slide_with_content(prs, title, sections, bg_color=(255, 255, 255)):
    """Add content slide with proper spacing"""
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    set_background_color(slide, bg_color)

    # Set title
    title_shape = slide.shapes.title
    title_shape.text = title
    title_shape.text_frame.paragraphs[0].font.size = Pt(24)  # Reduced to fit better

    # Get the content placeholder
    content_placeholder = slide.placeholders[1]
    tf = content_placeholder.text_frame
    tf.clear()
    tf.word_wrap = True

    # Add content with proper spacing
    for section_title, bullet_points in sections:
        # Add section title
        p = tf.add_paragraph()
        p.text = section_title
        p.font.size = Pt(16)  # Adjusted font size
        p.font.bold = True
        p.font.color.rgb = RGBColor(0, 0, 0)
        p.space_after = Pt(6)

        # Add bullet points
        for point in bullet_points:
            p = tf.add_paragraph()
            p.text = str(point)
            p.font.size = Pt(12)  # Adjusted font size
            p.level = 1
            p.font.color.rgb = RGBColor(0, 0, 0)
            p.space_after = Pt(3)

        # Add space between sections
        p = tf.add_paragraph()
        p.space_after = Pt(8)


def generate_direct_response(question: str, df: pd.DataFrame) -> str:
    """
    Generates a direct, conversational answer to a user's question based on the provided data.
    """
    # Create a prompt that instructs the LLM to answer the question
    prompt = f"Based on the following data ONLY, provide a direct answer to the question: '{question}'. Do not invent, add, or extrapolate new data. Use only the information in the provided data.\n\nData:\n{df.to_string(index=False)}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers user questions based on provided data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()
