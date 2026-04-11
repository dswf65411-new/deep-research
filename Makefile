.PHONY: init install-skill test clean

init:
	@bash setup.sh

install-skill:
	@bash setup.sh  # re-runs full setup, idempotent

test:
	@.venv/bin/python3 -c "\
		import sys; sys.path.insert(0,'.'); \
		from deep_research.graph import build_deep_research; \
		g = build_deep_research(); \
		print('Graph nodes:', list(g.get_graph().nodes.keys())); \
		print('✅ All OK')"

clean:
	rm -rf .venv __pycache__ deep_research/__pycache__ deep_research/**/__pycache__
