import streamlit as st

def feedback_page():
    st.title("Feedback")
    st.write(
        """
        We value your feedback! Reach out to us via email or join the discussion on GitHub. 
        Your thoughts help us improve and grow.
        """
    )
    st.markdown(
        """
        <div style="padding: 15px; border-radius: 10px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1); background: linear-gradient(135deg, #f0f2f5, #e9ebee);">
            <p style="font-size: 1.2rem; margin: 0; color: #333;">📧 
                <a href="mailto:ai_feedback@cloudera.com" style="text-decoration: none; color: #0078d4; font-weight: bold;">
                    ai_feedback@cloudera.com
                </a>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style="margin-top: 20px; padding: 15px; border-radius: 10px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1); background: linear-gradient(135deg, #e9f3ff, #dceaf8);">
            <p style="display: flex; align-items: center; font-size: 1.2rem; margin: 0; color: #333;">
                <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="30" style="margin-right: 12px; border-radius: 5px;" alt="GitHub Logo">
                <a href="https://github.com/cloudera/CML_AMP_LLM_Fine_Tuning_Studio/discussions" target="_blank" style="text-decoration: none; color: #0078d4; font-weight: bold;">
                    Join the discussion on GitHub
                </a>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

feedback_page()
