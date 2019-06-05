<template>
    <mdb-container>
        <mdb-row>
            <mdb-card class="text-center" style="height: auto; margin: 30px; width: fit-content">
                <mdb-card-header color="primary-color" tag="h3">Надетектило {{boxes.length}} людей! :)</mdb-card-header>
                <mdb-card>
                    <mdb-card-image id="image" :src="url" alt="Card image cap"></mdb-card-image>
                    <vue-draggable-resizable
                            v-if="loaded"
                            :active="true"
                            :prevent-deactivation="true"
                            :handles="['mr']"
                            :parent="true"
                            class="vdr"
                            class-name-handle="handler"
                            style="height: 100%; max-width: 100%;"
                            :draggable="false"
                    >
                        <div class="points">
                            <div v-for="(box, index) in boxes"
                                 :key="box.id"
                                 :style="{
                            'top': box[1] + 'px',
                            'left': box[0] + 'px',
                            'height': (box[3] - box[1]) + 'px',
                            'width': (box[2] - box[0]) + 'px',
                            'border-color': 'red'
                            }"
                                 class="point"
                            >{{(scores[index]).toFixed(2)}}</div>
                        </div>
                    </vue-draggable-resizable>
                </mdb-card>
                <mdb-btn color="success" @click="onClick">Задонатить!</mdb-btn>
            </mdb-card>
        </mdb-row>
    </mdb-container>
</template>

<script>
    import {mdbContainer, mdbRow, mdbCard, mdbCardHeader, mdbBtn, mdbCardImage} from 'mdbvue';
    export default {
        data () {
            return {
                boxes: [],
                scores: [],
                labels: [],
                loaded: false

            }
        },
        props : {
            url : {
                type: String
            },
            preds: {
              type: Object
            }
        },
        methods: {
            onClick() {
                this.$router.push({name: 'home'})
            },
            get_coords() {
                var img = document.getElementById('image');
                var rect=img.getBoundingClientRect();
                return rect
            },
            get_orig_shape() {
                var img = new Image();
                img.src = this.url;
                return {
                    'width': img.width,
                    'height': img.height
                }
            },
            rescale_boxes() {
                var rect = this.get_coords();
                var orig_shape = this.get_orig_shape();
                var k = rect.height/orig_shape.height;
                var true_width = orig_shape.width*rect.height/orig_shape.height;
                var x_shift = (rect.width - true_width)/2;
                this.boxes = this.boxes.map((box) => {
                    var new_box = [0,0,0,0];
                    new_box[0] = box[0]*k + x_shift;
                    new_box[1] = box[1]*k;
                    new_box[2] = box[2]*k + x_shift;
                    new_box[3] = box[3]*k;
                    return new_box
                });
            }
        },
        mounted () {
            this.boxes = this.preds['boxes'];
            this.scores = this.preds['scores'];
            this.labels = this.preds['labels'];

            var img = new Image();
            this.loaded = img.addEventListener('load', () => {
                this.loaded = true
                this.rescale_boxes();
            });
            img.src = this.url;
        },
        components: {
            mdbContainer,
            mdbRow,
            mdbCard,
            mdbCardHeader,
            mdbBtn,
            mdbCardImage
        }
    }
</script>

<style scoped>
    .vdr {
        position: relative;
        border-right: 10px solid #4285f4;
    }
    .points {
        position: absolute;
        top: 0;
        left: 0;
        width: inherit;
        height: inherit;
        overflow: hidden;
    }

    .point {
        position: absolute;
        width: 30px;
        height: 30px;
        border: 4px solid red;
    }
    .vdr >>> .handler {
        position: absolute;
        background-color: #4285f4;
        border: 2px solid #2b5ba6;
        border-radius: 50%;
        height: 25px;
        width: 25px;
        box-model: border-box;
        -webkit-transition: all 300ms linear;
        -ms-transition: all 300ms linear;
        transition: all 300ms linear;
    }

    .vdr >>> .handler-mr {
        top: 50%;
        margin-top: -7px;
        right: -17px;
        cursor: e-resize;
    }
</style>