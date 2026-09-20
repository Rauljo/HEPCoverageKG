"""The Streamlit demonstrator's helpers.

Kept out of `app.py` so the page stays a page. Nothing here is used by the
evaluation harness, and nothing here changes what a run does: these modules
call the same planner, free-SQL agent, chain and judge the experiments ran,
with the configuration the chapter kept, and add what the user sees on top --
provenance per paper, the paper-entity graph, and the conversation.
"""
