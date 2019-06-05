import VueRouter from 'vue-router'
import PredsView from './components/PredsView'
import Uploading from './components/Uploading'

export default new VueRouter({
    routes :[
        {
            path: '',
            name:'home',
            component: Uploading
        },
        {
            path: '/predictions',
            name: 'predictions',
            component: PredsView,
            props : true
        }
    ],
    mode: 'history'
})