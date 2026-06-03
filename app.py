import streamlit as st
import numpy as np
import time
import pandas as pd
import image_fusion as imfu
import graphs as grph
import glob
import create_save as creatsv
import cv2
import os

#Set page layout to wide to use max space
st.set_page_config(layout="wide",
                   initial_sidebar_state="collapsed"
)

#font weights
st.markdown("""
<style>
    /* Make all form labels bright and bold */
    label, p, th, td {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

#Background
st.html(f"""
    <style>
    .stApp {{
        background: linear-gradient(150deg, #00361d 0%, #000000 50%, #008247 100%);
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed !important; 
    }}
    </style>
""")

css = '''
<style>
    /* New Top Margin Fixes */
    .stAppViewMain > div {
        padding-top: 0rem;
    }
    [data-testid="stMainBlockContainer"] {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
    }

    [data-testid='stFileUploader'] {
        width: 100%;
    }
    [data-testid='stFileUploader'] section {
        padding: 0;
        width: 100%;
    }
    [data-testid='stFileUploader'] section > input + div {
        display: block;
    }
    [data-testid='stFileUploader'] section + div {
        float: right;
        padding-top: 0;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)

#Extra modifications to css
st.markdown(
    """
    <style>    
    [data-testid="stWidgetLabel"]{
        min-height: 0.5rem;
    }

    .st-key-main_box_left{
        position: relative;
    }

    .st-key-main_box_left [data-testid="stFileUploaderDropzone"]{
        background-color: #d3d3d3;
        padding: 0.3rem;
        min-height: 4rem;
        background-color: rgba(255, 255, 255, 0.15); 
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 2px dashed #5CE488;
        border-radius: 8px;
        opacity: 0.7;
    }

    .st-key-main_box_left [data-testid="stBaseButton-secondary"] {
        background-color: #0e1117;
        color: #5CE488;
        border: 1px solid;
    }

    .st-key-main_box_left [data-testid="stBaseButton-secondary"]:hover {
        background-color: #5CE488;
        color: #434445; 
    }
    
    div.stButton > button {
        background-color:#5CE488; 
        color: #434445;
        border: none;
        transition: background-color 0.2s ease !important; /* Smooth color change */
    }
    
    div.stButton > button:hover {
        background-color: #434445;
        color: #5CE488;
    }

    [data-testid="stImage"]{
        padding: 5px; 
        background-color: #0e1117;
        border-radius: 8px;
    }
    
    [data-baseweb="select"] > div {
        background-color: #0e1117;  
        color: #FFFFFF;            
        border-color: #5CE488;       
    }

    div[data-testid="stAlertContainer"] {
        background-color: #0e1117;
        color: #5CE488;
        width: 350px;
        height: 200px;
        min-height: 200px;
        display: flex;
        align-items: center;
    }

    [data-testid="stTable"] th, 
    [data-testid="stTable"] td{
        text-align: left !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)
#container colours
st.html("""
    <style>
    .st-key-main_box_left, .st-key-main_box_right{
            background: #2f3633;
            padding: 8px;
            border: 3px solid #0d6b1f;
            border-radius: 11px;
    }
    </style>
""")

left_main, right_main  = st.columns(2)
with left_main:
    with st.container(key ="main_box_left"):
        st.markdown(
            """
            <h5>
                <span style='color: #FFFFFF;'>IMAGE</span> 
                <span style='color: #0f9929;'>FUSION</span>
            </h5>
            """, 
            unsafe_allow_html=True
        )
        left, right = st.columns(2)
        if "fusion_method" not in st.session_state:
            st.session_state.fusion_method = "Patch based"
        with left:
            image_container = st.empty()
            dropbox_container = st.empty()  
            dropbox_container_colour = st.empty() 

            vis_paths = sorted(glob.glob("image_assets/vis_img/*"))
            vis_filenames = [path.replace("\\", "/").split("/")[-1] for path in vis_paths]

            selected_vis_filename = dropbox_container.selectbox(
                "Choose Vis Image:",
                vis_filenames if vis_filenames else ["No files found"],
                index=0,
                label_visibility="collapsed",
                key="dataset_vis_selector"
            )

            st.session_state.vis_file_path = selected_vis_filename

            if vis_filenames:
                vis_path = f"image_assets/vis_img/{selected_vis_filename}"
            else:
                vis_path = r"image_assets\FLIR_00006.jpg"
                
            vis_img = imfu.resizing(vis_path)
            
            current_method = st.session_state.get("fusion_method", None)

            chosen_vis_space = dropbox_container_colour.selectbox(
                "Choose vis image colour transform:",
                ["HSI", "LAB", "YCrCb", "CIE XYZ", "HLS", "YUV", "CIELUV"], 
                index=0, 
                label_visibility="collapsed",
                key="select_left", 
                filter_mode=None, #disables writing and search bar on select box
                #disables colour transform if not wavelet 
                disabled=current_method not in ["Wavelet + PCA", "Wavelet + Linear", "Wavelet + Salient"]
            )

            # 7. EXECUTE MATRIX MATH: Transform the array based on selected space
            # vis_intensity = imfu.get_Intensity(vis_img, chosen_vis_space)
            st.session_state.colour = chosen_vis_space
            image_container.image(
                vis_img, 
                caption=None, 
                width="content"
            )

        with right:
            image_container= st.empty()
            dropbox_container = st.empty()
            
            # 1. Look inside your IR subfolder
            ir_paths = sorted(glob.glob("image_assets/IR_img/*"))
            ir_filenames = [path.replace("\\", "/").split("/")[-1] for path in ir_paths]
            
            # 2. Render dropdown BELOW the image slot
            selected_ir_filename = dropbox_container.selectbox(
                "Choose IR Image:",
                ir_filenames if ir_filenames else ["No files found"],
                index=0,
                label_visibility="collapsed",
                key="dataset_ir_selector"
            )
            
            st.session_state.IR_file_path = selected_ir_filename

            if ir_filenames:
                ir_path = f"image_assets/IR_img/{selected_ir_filename}"
            else:
                ir_path = r"image_assets\FLIR_00006 (1).jpg"
            

            IR_img = imfu.resizing(ir_path)
            image_container.image(
                IR_img, 
                caption=None, 
                width="content"
            )
            #fusion method
            fusion_method = st.selectbox("Choose method of fusion",
                                        ["Patch based", "Wavelet + PCA", "Wavelet + Linear","Wavelet + Salient"],
                                        label_visibility="collapsed",
                                        filter_mode=None
                                        )

        level_container = st.empty()
        if "fused_output" in st.session_state and st.session_state.fused_output is not None:
            st.image(st.session_state.fused_output, 
                     width="content")
        else:
            st.info("Click 'FUSE IMAGES' below to generate the fused output.")
            
        # Trigger and each time its reruning
        if st.button("FUSE IMAGES", width="stretch"):
            start_time = time.time()
            current_level = st.session_state.get("level", 1)
            colour = st.session_state.get("colour", None) 
            if fusion_method == "Patch based":
                fused_result = imfu.overlapping_patch_based_image_fusion(vis_img, IR_img)
            elif fusion_method == "Wavelet + PCA":
                fused_result = imfu.Multilevel_Wavelet_PCA_CLAHE(vis_img, IR_img, colour, current_level)
            elif fusion_method == "Wavelet + Linear":
                fused_result = imfu.Multilevel_Wavelet_fusion(vis_img, IR_img, colour, current_level)
            elif fusion_method == "Wavelet + Salient":
                fused_result = imfu.wavelet_CLAHE_sal(vis_img, IR_img, colour, current_level)
            
            elapsed_time = time.time() - start_time

            hist_vis, hist_ir, hist_fused, metrics = grph.graph_entropy(vis_img, IR_img, fused_result)

            st.session_state.saved_vis = vis_img
            st.session_state.saved_IR = IR_img
            st.session_state.histv = hist_vis
            st.session_state.histi = hist_ir
            st.session_state.histf = hist_fused
            st.session_state.metrics = metrics
            st.session_state.fused_output = fused_result
            st.session_state.time_taken = elapsed_time
            st.session_state.fusion_method = fusion_method
            
            bgr_result = cv2.cvtColor(fused_result, cv2.COLOR_RGB2BGR)
            cv2.imwrite("fused.png", bgr_result)

            st.rerun()

        if fusion_method in ["Wavelet + PCA", "Wavelet + Linear","Wavelet + Salient"]:
            level = level_container.selectbox(
                "level decomposition",
                [1, 2, 3, 4],
                label_visibility="collapsed"
            )
            st.session_state.level = level


with right_main:
    with st.container(key="main_box_right", 
                    border=True):
        st.markdown(
            """
            <h5>
                <span style='color: #FFFFFF;'>Graph</span> 
                <span style='color: #0f9929;'>Image histogram</span>
            </h5>
            """, 
            unsafe_allow_html=True
        )    
        hist_vis = st.session_state.get("histv", None)
        hist_ir = st.session_state.get("histi", None)
        hist_fused = st.session_state.get("histf", None)

        #if no value yet before button
        if hist_vis is None or hist_ir is None or hist_fused is None:
            import numpy as np
            hist_vis = np.zeros(256)
            hist_ir = np.zeros(256)
            hist_fused = np.zeros(256)

        chart_data = pd.DataFrame({
            'Visible Image':hist_vis, 
            'IR Image':hist_ir, 
            'After Fusion':hist_fused
        })
        chart_data['Intensity'] = chart_data.index 
        cleaned_chart_data = chart_data.iloc[1:]
        st.line_chart(
            cleaned_chart_data,
            x='Intensity',
            y_label='Pixel Count',
            color=['#6FD600', '#008247', '#00ff2f'], 
            width="stretch"
        )
        
        fusion_method = st.session_state.get("fusion_method", None)
        fused_result = st.session_state.get("fused_output", None)
        
        data = {
            'Method': ['SF', 'Entropy', 'SSIM', 'StD', 'FQI', 'FSM', 'MI', 'Time taken'] 
        }

        fused_vals = [None]*8
 
        data1 = {
            'Method': ['SF', 'Entropy', 'SSIM', 'StD'] 
        }

        data2 = {
            'Method': ['FQI', 'FSM', 'MI', 'Time taken'] 
        }
        
        
        if fused_result is not None:  
            vis_img = st.session_state.get("saved_vis", vis_img)
            IR_img = st.session_state.get("saved_IR", IR_img)
            col1, col2 = st.columns(2)
        
            # 3. Render a dataframe/table in each column
            with col1:
                sf_val = grph.spatial_Freq(fused_result)
                entropy_val = grph.entropy(fused_result)
                ssim_val = grph.getSSISM(vis_img, fused_result)
                std_val = grph.Standard_Deviation(fused_result)

                fused_vals[0] = sf_val
                fused_vals[1] = entropy_val
                fused_vals[2] = ssim_val
                fused_vals[3] = std_val

                data1[fusion_method] = [sf_val, entropy_val, ssim_val, std_val]
                df1 = pd.DataFrame(data1)
                df_chart1 = df1.style.format(subset=[fusion_method], 
                                             formatter="{:.3f}")#3 decimal places displayed
                st.table(df_chart1)

            with col2:
                fqi_val, fsm_val = grph.FQIFSM(vis_img, IR_img, fused_result)
                mi_val = grph.mutual_information(vis_img, fused_result, bins=256)
                time_taken = st.session_state.get("time_taken", 0)

                fused_vals[4] = fqi_val
                fused_vals[5] = fsm_val
                fused_vals[6] = mi_val
                fused_vals[7] = time_taken

                data2[fusion_method] = [fqi_val, fsm_val, mi_val, time_taken]
                df2 = pd.DataFrame(data2)
                df_chart2 = df2.style.format(subset=[fusion_method], 
                                             formatter="{:.3f}")#3 decimal places displayed
                st.table(df_chart2)
        
        data[fusion_method] = fused_vals
        #clean it
        dataset = [["Metric", "Value"]]
        for name, value in zip(data['Method'], data[fusion_method]):
            dataset.append([name, str(value)])

        if "my_file_list" not in st.session_state:
            st.session_state.my_file_list = []

        @st.dialog("Save As")
        def save_popup(p_hvis, p_hir, p_hfused, p_d): 
            st.write("Where would you like to save this data ?")
            save_options = ["-- Choose an action --", "Create a new file"]+st.session_state.my_file_list
            choice = st.selectbox("file", 
                                  save_options, 
                                  label_visibility="collapsed",
                                  filter_mode= None
                                  )

            if choice == "Create a new file":
                total_files = len(st.session_state.my_file_list)
                next_number = total_files + 1
                name_of_file = f"report_part_imgfus{next_number}.pdf"
                
                # 1. Pull the simple string file paths chosen by the user
                vis_name = st.session_state.get('vis_file_path') 
                ir_name = st.session_state.get('IR_file_path') 
                fused_path = "fused.png"
                if vis_name and ir_name:
                    vis_path = os.path.join("image_assets", "vis_img", vis_name)
                    ir_path = os.path.join("image_assets", "IR_img", ir_name)

                    # 2. Pass the clean string paths directly to your PDF maker
                    st.download_button(
                        label="Download Fusion Report in .pdf",
                        data=creatsv.makepdf(vis_path, ir_path, fused_path, p_hvis, p_hir, p_hfused, p_d),
                        file_name=name_of_file,
                        mime="application/pdf"
                    )
                st.success(f"Successfully generated. file name: {name_of_file}")
                if name_of_file not in st.session_state.my_file_list:
                    st.session_state.my_file_list.append(name_of_file)
                    st.rerun() 
            elif "report_part_imgfus" in choice:
                vis_name = st.session_state.get('vis_file_path') 
                ir_name = st.session_state.get('IR_file_path') 
                fused_path = "fused.png"
                if vis_name and ir_name:
                    vis_path = os.path.join("image_assets", "vis_img", vis_name)
                    ir_path = os.path.join("image_assets", "IR_img", ir_name)

                    st.download_button(
                        label="Download Fusion Report in .pdf",
                        data=creatsv.makepdf(vis_path, ir_path, fused_path, p_hvis, p_hir, p_hfused, p_d),
                        file_name=choice,
                        mime="application/pdf"
                    )
                st.success(f"Successfully overwritten: {choice}")

        if st.button("PUBLISH", width="stretch"):
            # Only pass the numbers and metrics, keep images out of it
            save_popup(hist_vis, hist_ir, hist_fused, dataset)
