from streamlit.testing.v1 import AppTest

def test_streamlit_app():
    # Initialize AppTest with our app file
    at = AppTest.from_file("frontend/app.py").run()
    
    # 1. Verify title and subheader
    assert "Talking to Bridges" in at.title[0].value
    
    # 2. Verify sidebar upload functionality
    assert at.sidebar.file_uploader[0].label == "Upload a PDF, DOCX, or TXT file"
    assert at.sidebar.button[0].label == "Upload Document"
    
    # 3. Verify chat input
    assert at.chat_input[0] is not None
    
    print("Streamlit UI successfully loaded and passed basic layout tests!")

if __name__ == "__main__":
    test_streamlit_app()
