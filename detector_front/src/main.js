import Vue from 'vue'
import 'bootstrap-css-only/css/bootstrap.min.css';
import 'mdbvue/build/css/mdb.css';
import App from './App.vue'
import Router from 'vue-router'
import router from './routes'
import VueDraggableResizable from 'vue-draggable-resizable'


Vue.config.productionTip = false

Vue.use(Router);
export const eventEmitter = new Vue();

Vue.component('vue-draggable-resizable', VueDraggableResizable)
new Vue({
  render: h => h(App),
  router
}).$mount('#app')
