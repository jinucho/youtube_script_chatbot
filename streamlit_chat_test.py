import streamlit as st

id = "https://youtu.be/hrnVWt_x-LQ?si=KIgIMp_yqnGCpMMy".split("=")[-1]
st.write("video_id", id)

# st.markdown(
#     f'<iframe width="100%" height="600" src="https://www.youtube.com/embed/{id}" '
#     f'frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" '
#     f"allowfullscreen></iframe>",
#     unsafe_allow_html=True,
# )


st.markdown(
    """<iframe width="560" height="315" src="https://www.youtube.com/embed/HP7WYu_iigM?si=c-t9osWU9pnVHt60" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>""",
    unsafe_allow_html=True,
)
