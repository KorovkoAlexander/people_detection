IMAGE   := rambler-ai/sota-benchmark:$(shell id -nu)

all: build run

build:
	docker build --rm -t ${IMAGE} -f Dockerfile ..

run: things

things:
	docker run -it --rm --net=host --runtime=nvidia ${IMAGE}
