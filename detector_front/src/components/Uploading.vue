<template>
    <mdb-container>
        <mdb-row>
            <label class="btn btn-6 w-100 p-5 text-center">
                <h2>Загрузить фотографию</h2>
                <input type="file" hidden @change="handleFileUpload($event)">
            </label>
        </mdb-row>
    </mdb-container>
</template>

<script>
    import {mdbContainer, mdbRow} from 'mdbvue';
    import {apiRoot} from '../settings';
    import axios from 'axios';

    export default {
        name: 'Uploading',
        data () {
            return {
                image: '',
                imagePreview: '',
            }
        },
        methods: {
            get_predicts() {
                var bodyFormData = new FormData();
                bodyFormData.append('image', this.image);
                axios({
                    method: 'post',
                    url: 'http://5.228.220.51:8060/predict',
                    data: bodyFormData,
                    config: { headers: {'Content-Type': 'multipart/form-data' }}
                })
                    .then(response => {
                        this.$router.push({name: "predictions", params: {url: this.imagePreview, preds: response.data}})
                    })
                    .catch(e => {
                        this.$notify.error({message: 'Something went wrong :( Refresh!', position: 'top right', timeOut: 5000});
                    })
            },
            handleFileUpload(e) {
                e.preventDefault();
                this.image = e.target.files[0];
                let reader = new FileReader();

                reader.addEventListener("load", function () {
                    this.imagePreview = reader.result;
                }.bind(this), false);

                if (this.image) {
                    if (/\.(jpe?g|png|gif)$/i.test(this.image.name)) {
                        reader.readAsDataURL(this.image);
                    }
                }

                this.get_predicts()
            },
        },
        components: {
            mdbContainer,
            mdbRow,
        }
    }
</script>

<style scoped>
    label {
        margin-top: 30%;
        background-color: #4285F4;
    }
    h2 {
        color: white;
    }
</style>