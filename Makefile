
.PHONY: upload-code-cedar upload-code-narval upload-code-mila

noop:
	echo "preventing default action"

upload-code-cedar:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' --exclude='.git/' \
		-e ssh ./ cc-cedar:~/workspace/introspect

upload-code-narval:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' --exclude='.git/' \
		-e ssh ./ cc-narval:~/workspace/introspect

upload-code-mila:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' --exclude='.git/' \
		-e ssh ./ mila:~/workspace/introspect

download-database-mila:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' --exclude='.git/' \
		-e ssh mila:~/scratch/introspect/database/ ./database

download-database-narval:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' --exclude='.git/' \
		-e ssh cc-narval:~/scratch/introspect/database/ ./database

submitjobs:
	bash jobs/answerable.sh
	bash jobs/counterfactual.sh
	bash jobs/redacted.sh
