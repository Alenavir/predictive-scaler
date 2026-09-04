import http from 'k6/http';
import { sleep } from 'k6';

export const options = {
  summaryTrendStats: [
    'avg',
    'min',
    'med',
    'max',
    'p(95)',
    'p(99)',
  ],

  stages: [
    { duration: '1m', target: 20 },
    { duration: '2m', target: 150 },
    { duration: '1m', target: 30 },
    { duration: '2m', target: 400 },
    { duration: '1m', target: 20 },
    { duration: '3m', target: 0 },
  ],
};

export default function () {
  http.get('http://demo-app/');
  sleep(1);
}
