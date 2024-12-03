TARGET_FILES := $$(git diff --summary main quri_sdk_notebooks/ | sed {s/.*quri_sdk_notebooks/quri_sdk_notebooks/g})
TARGET_FILES_ALL := $$(find quri_sdk_notebooks -mindepth 1 -type f -name *.ipynb)
FILTER_STRING := (.cells[] | select(has("execution_count")) | .execution_count) = null | .metadata = {"language_info": {"name":"python", "pygments_lexer": "ipython3"}} | .cells[].metadata = {}


execute-in-place:
	for f in $(TARGET_FILES); do \
		poetry run jupyter execute --inplace --JupyterApp.log_level=50 $${f} ; \
	done


execute-in-place-all:
	for f in $(TARGET_FILES_ALL); do \
		poetry run jupyter execute --inplace --JupyterApp.log_level=50 $${f} ; \
	done


clean-nb:
	for f in $(TARGET_FILES_ALL); do \
		jq '$(FILTER_STRING)' "$${f}" > "$${f}.tmp" ; \
		mv "$${f}.tmp" "$${f}" ; \
	done
